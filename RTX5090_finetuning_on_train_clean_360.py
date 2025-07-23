#!/usr/bin/env python3
"""
Wav2Vec2 分布式训练脚本 - 修复freeze/unfreeze方法和检查点问题
"""

import os
import json
import torch
import torchaudio
import numpy as np
import socket
import time
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    get_cosine_schedule_with_warmup
)
import torch.nn.functional as F
from tqdm import tqdm
from jiwer import wer, cer
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from torch.cuda.amp import autocast, GradScaler
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import pickle
import functools
import gc
warnings.filterwarnings('ignore')

# 环境变量设置 - 优化分布式训练
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/fssd/ASR_task/.cache'
os.environ['HF_HOME'] = '/root/fssd/ASR_task/.cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 分布式优化设置
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

# 路径配置
BASE_PATH = "/root/fssd/ASR_task"
DATA_PATH = os.path.join(BASE_PATH, "huggingface/datasets/LibriSpeech")
SAVE_PATH = os.path.join(BASE_PATH, ".cache")
MODELS_PATH = os.path.join(BASE_PATH, ".cache/models")

# 训练历史记录类
class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_wers = []
        self.val_cers = []
        self.learning_rates = []
        self.epochs = []
        
    def update(self, epoch, train_loss, val_loss=None, val_wer=None, val_cer=None, lr=None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_wer is not None:
            self.val_wers.append(val_wer)
        if val_cer is not None:
            self.val_cers.append(val_cer)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def plot_curves(self, save_path):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.epochs, self.train_losses, 'b-', label='训练损失')
        if self.val_losses:
            axes[0, 0].plot(self.epochs[:len(self.val_losses)], self.val_losses, 'r-', label='验证损失')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # WER曲线
        if self.val_wers:
            axes[0, 1].plot(self.epochs[:len(self.val_wers)], self.val_wers, 'g-', label='验证WER')
        axes[0, 1].axhline(y=0.08, color='r', linestyle='--', label='目标WER (0.08)')
        axes[0, 1].set_title('词错误率 (WER)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # CER曲线
        if self.val_cers:
            axes[1, 0].plot(self.epochs[:len(self.val_cers)], self.val_cers, 'm-', label='验证CER')
        axes[1, 0].set_title('字符错误率 (CER)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率曲线
        if self.learning_rates:
            axes[1, 1].plot(self.epochs[:len(self.learning_rates)], self.learning_rates, 'c-', label='学习率')
            axes[1, 1].set_yscale('log')
        axes[1, 1].set_title('学习率调度')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_to_file(self, save_path):
        """保存历史记录到文件"""
        history_data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_wers': self.val_wers,
            'val_cers': self.val_cers,
            'learning_rates': self.learning_rates
        }
        history_file = os.path.join(save_path, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def load_from_file(self, save_path):
        """从文件加载历史记录"""
        history_file = os.path.join(save_path, 'training_history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history_data = json.load(f)
                self.epochs = history_data.get('epochs', [])
                self.train_losses = history_data.get('train_losses', [])
                self.val_losses = history_data.get('val_losses', [])
                self.val_wers = history_data.get('val_wers', [])
                self.val_cers = history_data.get('val_cers', [])
                self.learning_rates = history_data.get('learning_rates', [])

# 超参数配置 - 修复DDP问题
hyperparameters = {
    # 训练配置
    'num_epochs': 50,
    'batch_size_per_gpu': 8,
    'gradient_accumulation_steps': 16,
    'stage1_lr': 3e-5,
    'stage2_lr': 1e-5,
    'stage1_epochs': 15,
    'stage2_epochs': 35,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    
    # 数据增强
    'use_mixup': False,
    'speed_perturb': True,
    'speed_rates': [0.9, 1.0, 1.1],
    'noise_augment': True,
    'noise_levels': [0.001, 0.005],
    'noise_prob': 0.3,
    
    # Focal Loss - 修改为混合损失以确保所有参数被使用
    'use_focal_loss': True,
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,
    'focal_weight': 0.5,  # 新增：focal loss权重
    
    # 训练技巧
    'mixed_precision': True,
    'freeze_feature_extractor_epochs': 5,
    'layer_wise_lr': True,
    'layer_lr_decay': 0.95,
    
    # 分布式训练 - 修复DDP配置
    'world_size': 4,
    'backend': 'nccl',
    'find_unused_parameters': True,  # 关键修复：改为True
    'timeout_minutes': 30,
    
    # 数据配置
    'num_workers': 2,
    'prefetch_factor': 2,
    'seed': 42,
    'max_audio_length': 8.0,
    'sample_rate': 16000,
    'audio_normalize': True,
    'min_audio_length': 1.0,
    'max_target_length': 128,
    
    # 早停和检查点
    'early_stopping_patience': 10,
    'save_checkpoint_every': 5,
    'eval_frequency': 3,
    'logging_steps': 50,
    'resume_from_checkpoint': None,  # 新增：从检查点恢复
    
    # 保存路径
    'save_dir': os.path.join(SAVE_PATH, f"wav2vec2_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
}

def find_free_port():
    """查找可用端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def setup_distributed(rank, world_size, master_port=None):
    """稳定的分布式设置"""
    if master_port is None:
        master_port = find_free_port()
    
    # 设置环境变量
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # 添加更多稳定性设置
    os.environ['NCCL_TREE_THRESHOLD'] = '0'
    os.environ['NCCL_TIMEOUT'] = '1800'
    
    try:
        print(f"GPU {rank}: 初始化分布式，端口: {master_port}")
        
        # 使用更长的超时时间
        timeout = timedelta(minutes=hyperparameters['timeout_minutes'])
        
        # 重试机制
        max_retries = 5
        for retry in range(max_retries):
            try:
                dist.init_process_group(
                    backend='nccl',
                    init_method=f'env://',
                    rank=rank,
                    world_size=world_size,
                    timeout=timeout
                )
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"GPU {rank}: 初始化失败，重试 {retry + 1}/{max_retries}: {e}")
                    time.sleep(10)
                else:
                    raise e
        
        torch.cuda.set_device(rank)
        print(f"GPU {rank}: 分布式初始化成功!")
        
        # 简单的同步测试
        tensor = torch.tensor([rank], dtype=torch.float32, device=f'cuda:{rank}')
        dist.all_reduce(tensor)
        print(f"GPU {rank}: 同步测试通过，总和: {tensor.item()}")
        
        return True
        
    except Exception as e:
        print(f"GPU {rank}: 分布式初始化失败: {e}")
        return False

def cleanup_distributed():
    """安全的分布式清理"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            print("分布式已清理")
    except Exception as e:
        print(f"清理分布式时出错: {e}")

def is_main_process(rank):
    """检查是否为主进程"""
    return rank == 0

def freeze_feature_extractor(model):
    """冻结特征提取器"""
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = False

def unfreeze_feature_extractor(model):
    """解冻特征提取器"""
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = True

def collate_batch(batch, processor):
    """批处理函数"""
    if not batch or len(batch) == 0:
        return None
    
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    audio_arrays = [item["audio"] for item in batch]
    transcriptions = [item["transcription"] for item in batch]
    
    # 处理音频
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    # 处理标签
    with processor.as_target_processor():
        labels = processor(
            transcriptions,
            return_tensors="pt",
            padding=True
        )
    
    label_input_ids = labels["input_ids"]
    label_attention_mask = labels.attention_mask
    
    label_input_ids = label_input_ids.masked_fill(
        label_attention_mask.ne(1), -100
    )
    
    result = {
        "input_values": inputs.input_values,
        "labels": label_input_ids,
    }
    
    if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
        result["attention_mask"] = inputs.attention_mask
    else:
        batch_size, seq_len = inputs.input_values.shape
        result["attention_mask"] = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    return result

def focal_loss(logits, labels, gamma=2.0, alpha=0.25):
    """改进的Focal Loss实现"""
    batch_size = logits.size(0)
    vocab_size = logits.size(-1)
    
    min_seq_len = min(logits.size(1), labels.size(1))
    logits = logits[:, :min_seq_len, :]
    labels = labels[:, :min_seq_len]
    
    valid_mask = labels != -100
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    valid_mask_flat = valid_mask.reshape(-1)
    
    valid_logits = logits_flat[valid_mask_flat]
    valid_labels = labels_flat[valid_mask_flat]
    
    ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='none')
    p = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - p) ** gamma * ce_loss
    
    return focal_loss.mean()

class LibriSpeechDataset(Dataset):
    def __init__(self, data_dirs, processor, hyperparams, is_train=True, rank=0):
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.processor = processor
        self.hyperparams = hyperparams
        self.is_train = is_train
        self.rank = rank
        
        self.max_audio_length = hyperparams['max_audio_length']
        self.min_audio_length = hyperparams['min_audio_length']
        self.sample_rate = hyperparams['sample_rate']
        self.audio_normalize = hyperparams['audio_normalize']
        self.max_target_length = hyperparams['max_target_length']
        self.max_length_samples = int(self.max_audio_length * self.sample_rate)
        self.min_length_samples = int(self.min_audio_length * self.sample_rate)
        
        self.audio_files = []
        self.transcriptions = []
        
        # 缓存文件
        cache_name = f"dataset_cache_{'-'.join([os.path.basename(d) for d in self.data_dirs])}.pkl"
        self.cache_file = os.path.join(SAVE_PATH, cache_name)
        
        if os.path.exists(self.cache_file):
            self._load_from_cache()
        else:
            self._load_data()
            self._save_to_cache()
        
        if len(self.audio_files) == 0:
            raise ValueError(f"没有找到任何音频文件！请检查路径: {self.data_dirs}")
    
    def _load_from_cache(self):
        """从缓存加载数据"""
        if is_main_process(self.rank):
            print(f"从缓存加载数据: {self.cache_file}")
        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            self.audio_files = cache_data['audio_files']
            self.transcriptions = cache_data['transcriptions']
    
    def _save_to_cache(self):
        """保存数据到缓存"""
        if is_main_process(self.rank):
            print(f"保存数据到缓存: {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'audio_files': self.audio_files,
                    'transcriptions': self.transcriptions,
                }, f)
    
    def _load_data(self):
        """加载数据"""
        if is_main_process(self.rank):
            print(f"GPU {self.rank}: 开始加载数据...")
        
        count = 0
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                if is_main_process(self.rank):
                    print(f"警告: 路径不存在 {data_dir}")
                continue
            
            if is_main_process(self.rank):
                print(f"处理数据目录: {data_dir}")
            
            for root, dirs, files in os.walk(data_dir):
                trans_files = [f for f in files if f.endswith('.trans.txt')]
                
                for trans_file in trans_files:
                    trans_path = os.path.join(root, trans_file)
                    
                    try:
                        with open(trans_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split(' ', 1)
                            if len(parts) == 2:
                                audio_id, transcription = parts
                                audio_file = os.path.join(root, f"{audio_id}.flac")
                                
                                if os.path.exists(audio_file) and transcription.strip():
                                    clean_transcription = ' '.join(transcription.upper().split())
                                    
                                    # 检查转录长度
                                    try:
                                        encoded_length = len(self.processor.tokenizer.encode(clean_transcription))
                                        if encoded_length > self.max_target_length:
                                            continue
                                    except:
                                        continue
                                    
                                    # 检查音频长度
                                    try:
                                        info = torchaudio.info(audio_file)
                                        duration = info.num_frames / info.sample_rate
                                        
                                        if (duration >= self.min_audio_length and 
                                            duration <= self.max_audio_length):
                                            
                                            self.audio_files.append(audio_file)
                                            self.transcriptions.append(clean_transcription)
                                            count += 1
                                            
                                            if count % 1000 == 0 and is_main_process(self.rank):
                                                print(f"已加载 {count} 个文件...")
                                    except:
                                        continue
                    except Exception as e:
                        if is_main_process(self.rank):
                            print(f"读取转录文件出错 {trans_path}: {e}")
                        continue
        
        if is_main_process(self.rank):
            print(f"数据加载完成，总样本数: {len(self.audio_files)}")
    
    def _normalize_audio(self, audio_array):
        """音频标准化"""
        if self.audio_normalize:
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
            rms = np.sqrt(np.mean(audio_array ** 2))
            if rms > 1e-8:
                target_rms = 0.15
                audio_array = audio_array * (target_rms / rms)
            audio_array = np.clip(audio_array, -1.0, 1.0)
        return audio_array.astype(np.float32)
    
    def _apply_augmentation(self, waveform):
        """数据增强"""
        if not self.is_train:
            return waveform
        
        # 速度扰动
        if self.hyperparams['speed_perturb'] and random.random() < 0.3:
            speed_rate = random.choice(self.hyperparams['speed_rates'])
            if speed_rate != 1.0:
                try:
                    effects = [["speed", str(speed_rate)], ["rate", str(self.sample_rate)]]
                    waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                        waveform, self.sample_rate, effects
                    )
                except:
                    pass
        
        # 添加噪声
        if self.hyperparams['noise_augment'] and random.random() < self.hyperparams['noise_prob']:
            noise_level = random.choice(self.hyperparams['noise_levels'])
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise
        
        return waveform
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        transcription = self.transcriptions[idx]
        
        try:
            # 加载音频
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # 重采样
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # 转为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 数据增强
            waveform = self._apply_augmentation(waveform)
            
            # 转换为numpy
            audio_array = waveform.squeeze().numpy()
            
            # 长度处理
            if len(audio_array) > self.max_length_samples:
                if self.is_train:
                    start = np.random.randint(0, len(audio_array) - self.max_length_samples + 1)
                else:
                    start = (len(audio_array) - self.max_length_samples) // 2
                audio_array = audio_array[start:start + self.max_length_samples]
            elif len(audio_array) < self.min_length_samples:
                audio_array = np.pad(audio_array, (0, self.min_length_samples - len(audio_array)))
            
            # 标准化
            audio_array = self._normalize_audio(audio_array)
            
            return {
                "audio": audio_array,
                "transcription": transcription,
            }
            
        except Exception as e:
            print(f"加载音频出错 {audio_file}: {e}")
            return self.__getitem__((idx + 1) % len(self))

def create_dataloaders(processor, hyperparams, rank, world_size):
    """创建数据加载器"""
    # 数据路径
    train_dirs = []
    val_dirs = []
    
    # 训练集
    train_clean_100 = os.path.join(DATA_PATH, "train-clean-360")
    if os.path.exists(train_clean_100):
        train_dirs.append(train_clean_100)
    
    # 验证集
    dev_clean = os.path.join(DATA_PATH, "dev-clean")
    if os.path.exists(dev_clean):
        val_dirs.append(dev_clean)
    
    if not train_dirs:
        raise ValueError(f"没有找到训练数据集！请检查路径: {DATA_PATH}")
    if not val_dirs:
        val_dirs = [train_dirs[0]]
    
    if is_main_process(rank):
        print(f"训练集路径: {train_dirs}")
        print(f"验证集路径: {val_dirs}")
    
    # 创建数据集
    train_dataset = LibriSpeechDataset(train_dirs, processor, hyperparams, is_train=True, rank=rank)
    val_dataset = LibriSpeechDataset(val_dirs, processor, hyperparams, is_train=False, rank=rank)
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=hyperparams['seed'])
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=hyperparams['seed'])
    
    # 创建collate函数
    collate_fn = functools.partial(collate_batch, processor=processor)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=hyperparams['batch_size_per_gpu'], sampler=train_sampler,
        num_workers=hyperparams['num_workers'], pin_memory=True, collate_fn=collate_fn, drop_last=True,
        prefetch_factor=hyperparams['prefetch_factor'] if hyperparams['num_workers'] > 0 else None
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=hyperparams['batch_size_per_gpu'], sampler=val_sampler,
        num_workers=hyperparams['num_workers'], pin_memory=True, collate_fn=collate_fn, drop_last=False,
        prefetch_factor=hyperparams['prefetch_factor'] if hyperparams['num_workers'] > 0 else None
    )
    
    return train_dataloader, val_dataloader, train_dataset, val_dataset

def evaluate_model(model, dataloader, processor, device, rank):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_references = []
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="评估中", disable=not is_main_process(rank))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue
                
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # 解码预测
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            # 解码参考文本
            labels_copy = labels.clone()
            labels_copy[labels_copy == -100] = processor.tokenizer.pad_token_id
            reference_texts = processor.batch_decode(labels_copy, skip_special_tokens=True)
            
            all_predictions.extend(predicted_texts)
            all_references.extend(reference_texts)
            
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
    
    # 计算指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    wer_score = wer(all_references, all_predictions)
    cer_score = cer(all_references, all_predictions)
    
    return avg_loss, wer_score, cer_score, all_predictions, all_references

def train_epoch(model, dataloader, optimizer, scheduler, scaler, processor, device, epoch, hyperparams, rank):
    """训练一个epoch - 修复参数未使用问题"""
    model.train()
    total_loss = 0
    num_steps = 0
    
    if len(dataloader) == 0:
        print(f"GPU {rank}: 警告 - 数据加载器为空！")
        return 0.0
    
    # 设置随机种子
    torch.manual_seed(hyperparams['seed'] + epoch)
    np.random.seed(hyperparams['seed'] + epoch)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process(rank))
    
    for step, batch in enumerate(progress_bar):
        try:
            if batch is None:
                continue
                
            # 准备数据
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 混合精度训练
            with autocast(enabled=hyperparams['mixed_precision']):
                # 前向传播 - 始终使用模型的输出
                outputs = model(
                    input_values=input_values, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                # 混合损失策略：结合模型损失和focal loss
                if hyperparams['use_focal_loss'] and hasattr(outputs, 'logits'):
                    # 获取模型原生损失（确保所有参数参与计算）
                    model_loss = outputs.loss
                    
                    # 计算focal loss
                    f_loss = focal_loss(
                        outputs.logits, 
                        labels, 
                        gamma=hyperparams['focal_gamma'], 
                        alpha=hyperparams['focal_alpha']
                    )
                    
                    # 混合两种损失
                    focal_weight = hyperparams['focal_weight']
                    loss = (1 - focal_weight) * model_loss + focal_weight * f_loss
                else:
                    # 只使用模型损失
                    loss = outputs.loss
                
                # 梯度累积
                loss = loss / hyperparams['gradient_accumulation_steps']
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积和更新
            if (step + 1) % hyperparams['gradient_accumulation_steps'] == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['max_grad_norm'])
                
                # 优化器步骤
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # 学习率调度
                if scheduler is not None:
                    scheduler.step()
                
                num_steps += 1
            
            # 记录损失
            total_loss += loss.item() * hyperparams['gradient_accumulation_steps']
            
            # 更新进度条
            if is_main_process(rank):
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * hyperparams['gradient_accumulation_steps']:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            # 日志记录
            if step % hyperparams['logging_steps'] == 0 and is_main_process(rank):
                current_loss = loss.item() * hyperparams['gradient_accumulation_steps']
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step}, Loss: {current_loss:.4f}, LR: {current_lr:.2e}")
            
            # 定期清理缓存
            if step % 10 == 0:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "find_unused_parameters" in str(e):
                print(f"GPU {rank}: DDP参数未使用错误，尝试继续...")
                # 清理状态
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        except Exception as e:
            print(f"GPU {rank}: 训练步骤 {step} 出错: {e}")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
    
    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss

def get_layer_wise_lr(model, base_lr, layer_decay):
    """获取层级学习率"""
    parameters = []
    
    # 特征提取器
    parameters.append({
        'params': model.wav2vec2.feature_extractor.parameters(),
        'lr': base_lr * (layer_decay ** 3)
    })
    
    # 特征投影
    parameters.append({
        'params': model.wav2vec2.feature_projection.parameters(),
        'lr': base_lr * (layer_decay ** 2)
    })
    
    # Transformer层
    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        parameters.append({
            'params': layer.parameters(),
            'lr': base_lr * (layer_decay ** (len(model.wav2vec2.encoder.layers) - i))
        })
    
    # 输出层
    parameters.append({
        'params': model.lm_head.parameters(),
        'lr': base_lr
    })
    
    return parameters

def save_checkpoint(model, optimizer, scheduler, epoch, best_wer, checkpoint_path, scaler=None, history=None):
    """改进的检查点保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_wer': best_wer,
        'history': history.__dict__ if history else None,
        'hyperparameters': hyperparameters
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # 使用临时文件防止保存过程中断
    temp_path = checkpoint_path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.rename(temp_path, checkpoint_path)
    print(f"检查点已保存: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """改进的检查点加载"""
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0, float('inf'), None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型状态
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载梯度缩放器状态
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 加载历史记录
        history = None
        if 'history' in checkpoint and checkpoint['history']:
            history = TrainingHistory()
            history.__dict__.update(checkpoint['history'])
        
        epoch = checkpoint.get('epoch', 0)
        best_wer = checkpoint.get('best_wer', float('inf'))
        
        print(f"成功加载检查点: {checkpoint_path}")
        print(f"恢复到 epoch {epoch}, 最佳WER: {best_wer:.4f}")
        
        return epoch, best_wer, history
        
    except Exception as e:
        print(f"加载检查点失败 {checkpoint_path}: {e}")
        return 0, float('inf'), None

def find_latest_checkpoint(save_dir):
    """查找最新的检查点"""
    if not os.path.exists(save_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(save_dir):
        if file.startswith('checkpoint_epoch_') and file.endswith('.pth'):
            try:
                epoch_num = int(file.split('_')[2].split('.')[0])
                checkpoint_files.append((epoch_num, os.path.join(save_dir, file)))
            except:
                continue
    
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: x[0])
        return latest_checkpoint[1]
    
    # 检查是否有best_model.pth
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        return best_model_path
    
    return None

def load_model_and_processor(rank=0):
    """统一的模型和处理器加载函数"""
    possible_paths = [
        "/root/fssd/ASR_task/.cache/wav2vec2_finetuned_20250721_015618/best_finetuned_model",
        os.path.join(MODELS_PATH, "wav2vec2-large-960h"),
        "facebook/wav2vec2-large-960h"
    ]
    
    model = None
    processor = None
    loaded_from = None
    
    for path in possible_paths:
        try:
            if is_main_process(rank):
                print(f"GPU {rank}: 尝试从 {path} 加载模型...")
            
            if os.path.exists(path) and os.path.isdir(path):
                # 本地路径
                model = Wav2Vec2ForCTC.from_pretrained(path, ctc_loss_reduction="mean", pad_token_id=0)
                processor = Wav2Vec2Processor.from_pretrained(path)
                loaded_from = f"本地路径: {path}"
            elif not os.path.exists(path):
                # HuggingFace路径
                model = Wav2Vec2ForCTC.from_pretrained(path, ctc_loss_reduction="mean", pad_token_id=0, cache_dir=MODELS_PATH)
                processor = Wav2Vec2Processor.from_pretrained(path, cache_dir=MODELS_PATH)
                loaded_from = f"HuggingFace: {path}"
            
            if model is not None and processor is not None:
                if is_main_process(rank):
                    print(f"GPU {rank}: ✅ 成功加载模型，来源: {loaded_from}")
                break
                
        except Exception as e:
            if is_main_process(rank):
                print(f"GPU {rank}: ❌ 从 {path} 加载失败: {e}")
            continue
    
    if model is None or processor is None:
        raise ValueError("❌ 无法从任何路径加载模型！请检查模型文件或网络连接。")
    
    return model, processor, loaded_from

def train_model(rank, world_size, hyperparams, master_port):
    """主训练函数 - 修复freeze/unfreeze和检查点问题"""
    try:
        # 设置分布式
        success = setup_distributed(rank, world_size, master_port)
        if not success:
            return
        
        device = torch.device(f'cuda:{rank}')
        
        # 设置随机种子
        seed = hyperparams['seed'] + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if is_main_process(rank):
            print("开始训练流程...")
        
        # 加载模型和处理器
        model, processor, loaded_from = load_model_and_processor(rank)
        model.to(device)
        
        # 创建DDP模型 - 使用find_unused_parameters=True
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=hyperparams['find_unused_parameters'],  # 使用True
            broadcast_buffers=True,
            bucket_cap_mb=25,
            gradient_as_bucket_view=True
        )
        
        if is_main_process(rank):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✅ 模型加载成功，来源: {loaded_from}")
            print(f"模型总参数数量: {total_params:,}")
            print(f"可训练参数数量: {trainable_params:,}")
            print(f"DDP配置: find_unused_parameters={hyperparams['find_unused_parameters']}")
        
        # 创建数据加载器
        train_dataloader, val_dataloader, train_dataset, val_dataset = create_dataloaders(processor, hyperparams, rank, world_size)
        
        if is_main_process(rank):
            print(f"训练集大小: {len(train_dataset)}")
            print(f"验证集大小: {len(val_dataset)}")
        
        # 创建优化器
        if hyperparams['layer_wise_lr']:
            parameters = get_layer_wise_lr(model.module, hyperparams['stage1_lr'], hyperparams['layer_lr_decay'])
            optimizer = AdamW(parameters, weight_decay=hyperparams['weight_decay'], eps=1e-8)
        else:
            optimizer = AdamW(model.parameters(), lr=hyperparams['stage1_lr'], weight_decay=hyperparams['weight_decay'], eps=1e-8)
        
        # 学习率调度器
        num_training_steps = (len(train_dataloader) * hyperparams['num_epochs'] // hyperparams['gradient_accumulation_steps'])
        num_warmup_steps = int(0.1 * num_training_steps)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=0.5, last_epoch=-1
        )
        
        # 混合精度训练
        scaler = GradScaler(enabled=hyperparams['mixed_precision'])
        
        # 初始化训练变量
        start_epoch = 0
        best_wer = float('inf')
        patience_counter = 0
        history = TrainingHistory() if is_main_process(rank) else None
        
        # 创建保存目录
        if is_main_process(rank):
            os.makedirs(hyperparams['save_dir'], exist_ok=True)
        
        # 同步所有进程
        if dist.is_initialized():
            dist.barrier()
        
        # 检查是否需要从检查点恢复
        resume_checkpoint = hyperparams.get('resume_from_checkpoint')
        if resume_checkpoint:
            if os.path.exists(resume_checkpoint):
                checkpoint_path = resume_checkpoint
            else:
                checkpoint_path = find_latest_checkpoint(hyperparams['save_dir'])
        else:
            checkpoint_path = find_latest_checkpoint(hyperparams['save_dir'])
        
        if checkpoint_path and is_main_process(rank):
            print(f"发现检查点: {checkpoint_path}")
            start_epoch, best_wer, loaded_history = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler
            )
            if loaded_history and history:
                history = loaded_history
            start_epoch += 1  # 从下一个epoch开始
        
        # 加载历史记录
        if history and is_main_process(rank):
            history.load_from_file(hyperparams['save_dir'])
        
        if is_main_process(rank):
            print(f"训练配置:")
            print(f"  - 起始epoch: {start_epoch}")
            print(f"  - 总训练步数: {num_training_steps}")
            print(f"  - 预热步数: {num_warmup_steps}")
            print(f"  - 批次大小/GPU: {hyperparams['batch_size_per_gpu']}")
            print(f"  - 梯度累积步数: {hyperparams['gradient_accumulation_steps']}")
            print(f"  - 有效批次大小: {hyperparams['batch_size_per_gpu'] * hyperparams['gradient_accumulation_steps'] * world_size}")
            print(f"  - 当前最佳WER: {best_wer:.4f}")
        
        # 训练循环
        for epoch in range(start_epoch, hyperparams['num_epochs']):
            # 更新采样器的epoch
            train_dataloader.sampler.set_epoch(epoch)
            if val_dataloader.sampler:
                val_dataloader.sampler.set_epoch(epoch)
            
            # 阶段性学习率调整
            if epoch == hyperparams['stage1_epochs']:
                if is_main_process(rank):
                    print(f"切换到第二阶段学习率: {hyperparams['stage2_lr']}")
                for param_group in optimizer.param_groups:
                    if hyperparams['layer_wise_lr']:
                        param_group['lr'] = param_group['lr'] * (hyperparams['stage2_lr'] / hyperparams['stage1_lr'])
                    else:
                        param_group['lr'] = hyperparams['stage2_lr']
            
            # 冻结/解冻特征提取器 - 修复方法调用
            if epoch < hyperparams['freeze_feature_extractor_epochs']:
                freeze_feature_extractor(model.module)
            else:
                unfreeze_feature_extractor(model.module)
            
            # 训练
            if is_main_process(rank):
                print(f"\n===== Epoch {epoch + 1}/{hyperparams['num_epochs']} =====")
            
            train_loss = train_epoch(
                model, train_dataloader, optimizer, scheduler, scaler,
                processor, device, epoch, hyperparams, rank
            )
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 评估
            should_evaluate = ((epoch + 1) % hyperparams['eval_frequency'] == 0 or epoch == hyperparams['num_epochs'] - 1)
            
            if should_evaluate:
                if is_main_process(rank):
                    print("开始验证...")
                
                val_loss, val_wer, val_cer, _, _ = evaluate_model(model.module, val_dataloader, processor, device, rank)
                
                if is_main_process(rank):
                    print(f"Epoch {epoch + 1}:")
                    print(f"  训练损失: {train_loss:.4f}")
                    print(f"  验证损失: {val_loss:.4f}")
                    print(f"  验证WER: {val_wer:.4f}")
                    print(f"  验证CER: {val_cer:.4f}")
                    
                    # 更新历史记录
                    history.update(epoch + 1, train_loss, val_loss, val_wer, val_cer, current_lr)
                    history.plot_curves(hyperparams['save_dir'])
                    history.save_to_file(hyperparams['save_dir'])
                
                # 保存最佳模型
                is_best = val_wer < best_wer
                if is_best:
                    best_wer = val_wer
                    patience_counter = 0
                    
                    if is_main_process(rank):
                        print(f"新的最佳WER: {best_wer:.4f}")
                        
                        # 保存最佳模型检查点
                        best_model_path = os.path.join(hyperparams['save_dir'], 'best_model.pth')
                        save_checkpoint(model, optimizer, scheduler, epoch, best_wer, best_model_path, scaler, history)
                        
                        # 保存模型和处理器（用于推理）
                        best_model_dir = os.path.join(hyperparams['save_dir'], 'best_finetuned_model')
                        os.makedirs(best_model_dir, exist_ok=True)
                        model.module.save_pretrained(best_model_dir)
                        processor.save_pretrained(best_model_dir)
                        
                        # 保存训练配置
                        config_path = os.path.join(hyperparams['save_dir'], 'training_config.json')
                        with open(config_path, 'w') as f:
                            config_to_save = hyperparams.copy()
                            config_to_save['save_dir'] = str(config_to_save['save_dir'])
                            json.dump(config_to_save, f, indent=2, default=str)
                else:
                    patience_counter += 1
                
                # 定期保存检查点
                if (epoch + 1) % hyperparams['save_checkpoint_every'] == 0 and is_main_process(rank):
                    checkpoint_path = os.path.join(hyperparams['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
                    save_checkpoint(model, optimizer, scheduler, epoch, best_wer, checkpoint_path, scaler, history)
            
            else:
                # 即使不评估也要更新历史记录
                if is_main_process(rank):
                    history.update(epoch + 1, train_loss, lr=current_lr)
                    history.save_to_file(hyperparams['save_dir'])
            
            # 早停检查
            if patience_counter >= hyperparams['early_stopping_patience']:
                if is_main_process(rank):
                    print(f"早停触发！连续 {patience_counter} 轮没有改善")
                break
            
            # 垃圾回收
            if (epoch + 1) % 3 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # 训练结束
        if is_main_process(rank):
            print("\n" + "="*50)
            print("训练完成!")
            print(f"最佳验证WER: {best_wer:.4f}")
            print(f"所有结果已保存到: {hyperparams['save_dir']}")
            print("="*50)
    
    except Exception as e:
        print(f"GPU {rank}: 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        if 'model' in locals():
            del model
        
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 清理分布式
        cleanup_distributed()
        
        if is_main_process(rank):
            print(f"GPU {rank}: 资源清理完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Wav2Vec2 分布式训练')
    parser.add_argument('--world_size', type=int, default=4, help='GPU数量')
    parser.add_argument('--master_port', type=int, default=None, help='主节点端口')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--save_dir', type=str, default=None, help='保存目录')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 加载配置
    hyperparams = hyperparameters.copy()
    
    # 从命令行参数更新
    if args.epochs is not None:
        hyperparams['num_epochs'] = args.epochs
    if args.batch_size is not None:
        hyperparams['batch_size_per_gpu'] = args.batch_size
    if args.lr is not None:
        hyperparams['stage1_lr'] = args.lr
    if args.save_dir is not None:
        hyperparams['save_dir'] = args.save_dir
    if args.resume_from_checkpoint is not None:
        hyperparams['resume_from_checkpoint'] = args.resume_from_checkpoint
    
    # 更新world_size
    world_size = args.world_size
    hyperparams['world_size'] = world_size
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用！请确保有可用的GPU。")
    
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        print(f"警告: 请求的GPU数量({world_size})大于可用GPU数量({available_gpus})，调整为{available_gpus}")
        world_size = available_gpus
        hyperparams['world_size'] = world_size
    
    print(f"开始训练，使用 {world_size} 个GPU")
    print(f"训练配置:")
    print(f"  - 训练轮数: {hyperparams['num_epochs']}")
    print(f"  - 批次大小/GPU: {hyperparams['batch_size_per_gpu']}")
    print(f"  - 初始学习率: {hyperparams['stage1_lr']}")
    print(f"  - 保存目录: {hyperparams['save_dir']}")
    if hyperparams.get('resume_from_checkpoint'):
        print(f"  - 恢复检查点: {hyperparams['resume_from_checkpoint']}")
    
    # 创建保存目录
    os.makedirs(hyperparams['save_dir'], exist_ok=True)
    
    # 保存初始配置
    config_path = os.path.join(hyperparams['save_dir'], 'initial_config.json')
    with open(config_path, 'w') as f:
        json.dump(hyperparams, f, indent=2, default=str)
    
    # 查找可用端口
    master_port = args.master_port
    if master_port is None:
        master_port = find_free_port()
    
    print(f"使用端口: {master_port}")
    
    # 设置多进程启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 启动分布式训练
    try:
        if world_size == 1:
            print("启动单GPU训练...")
            train_model(0, 1, hyperparams, master_port)
        else:
            print(f"启动多GPU分布式训练，使用 {world_size} 个GPU...")
            mp.spawn(train_model, args=(world_size, hyperparams, master_port), nprocs=world_size, join=True)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("训练流程结束")
    return 0

if __name__ == "__main__":
    import sys
    
    # 检查CUDA
    try:
        if not torch.cuda.is_available():
            print("\n⚠️  警告: CUDA不可用，将无法使用GPU训练")
            response = input("是否继续? (y/n): ").lower()
            if response != 'y':
                sys.exit(0)
        else:
            print(f"\n✅ 检测到 {torch.cuda.device_count()} 个可用GPU")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"\n❌ CUDA检查失败: {e}")
        sys.exit(1)
    
    print("\n🚀 启动训练程序...\n")
    
    # 运行主程序
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ 程序异常退出: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)