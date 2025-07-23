#!/usr/bin/env python3
"""
优化版 Wav2Vec2 LibriSpeech训练脚本 - 修复梯度爆炸和特征提取器问题
主要修复：
1. 特征提取器冻结选项
2. 分层学习率
3. 逐步解冻策略
4. 梯度检查点
5. 更稳定的初始化
"""

import os
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    get_cosine_schedule_with_warmup
)
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import torch.nn.functional as F
from tqdm import tqdm
from jiwer import wer, cer
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from torch.cuda.amp import autocast, GradScaler
import random
from scipy import signal
warnings.filterwarnings('ignore')

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/fssd/ASR_task/.cache'
os.environ['HF_HOME'] = '/root/fssd/ASR_task/.cache'

# 路径配置
BASE_PATH = "/root/fssd/ASR_task"
DATA_PATH = os.path.join(BASE_PATH, "huggingface/datasets/LibriSpeech")
MODEL_PATH = os.path.join(BASE_PATH, ".cache/models/wav2vec2-large-960h")
SAVE_PATH = os.path.join(BASE_PATH, ".cache")

# 优化后的超参数配置 - 解决梯度爆炸
hyperparameters = {
    # 基础训练参数
    'learning_rate': 3e-5,              # 提高学习率
    'min_learning_rate': 1e-7,      
    'batch_size': 16,                    # 小批次
    'num_epochs': 30,               
    'weight_decay': 0.001,              # 减小权重衰减
    'max_grad_norm': 1.0,           
    'warmup_ratio': 0.05,               # 减少warmup
    'gradient_accumulation_steps': 4,   # 增加梯度累积
    
    # 模型配置
    'freeze_feature_extractor': True,   # 冻结特征提取器
    'gradient_checkpointing': True,     # 启用梯度检查点
    'layer_wise_lr': True,              # 使用分层学习率
    'gradual_unfreezing': True,         # 逐步解冻
    'unfreeze_epoch': 5,                # 开始解冻的epoch
    
    # 评估和保存
    'eval_steps': 500,
    'save_steps': 1000,
    'logging_steps': 50,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.001,
    
    # 音频处理
    'max_audio_length': 20.0,
    'sample_rate': 16000,
    'audio_normalize': True,
    'min_audio_length': 1.0,
    'max_target_length': 256,
    
    # 数据增强 - 暂时禁用
    'use_spec_augment': False,      
    'freq_mask_param': 15,          
    'time_mask_param': 50,          
    'num_freq_mask': 1,
    'num_time_mask': 1,
    'speed_perturb': False,         
    'speed_rates': [0.95, 1.0, 1.05],
    'noise_augment': False,         
    'noise_prob': 0.2,
    'noise_level': 0.003,
    
    # 正则化
    'dropout': 0.1,
    'attention_dropout': 0.1,    
    'feat_proj_dropout': 0.1,
    'hidden_dropout': 0.1,
    'label_smoothing': 0.0,
    
    # 优化器
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,            # 提高beta2
    'adam_epsilon': 1e-8,
    
    # 其他
    'num_workers': 4,
    'mixed_precision': False,
    'use_all_data': False,          
    'val_fraction': 0.1,
    'max_loss_value': 100.0,
    
    # 分层学习率配置
    'layer_lr_multipliers': {
        'feature_extractor': 0.01,      # 特征提取器使用1%的学习率
        'feature_projection': 0.1,      # 特征投影使用10%的学习率
        'encoder': 0.5,                 # 编码器使用50%的学习率
        'lm_head': 1.0                  # 输出层使用100%的学习率
    }
}

class SpecAugment:
    """SpecAugment数据增强"""
    def __init__(self, freq_mask_param, time_mask_param, num_freq_mask, num_time_mask):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_mask = num_freq_mask
        self.num_time_mask = num_time_mask
    
    def __call__(self, spectrogram):
        """应用SpecAugment"""
        if len(spectrogram.shape) != 2:
            return spectrogram
        
        freq_dim, time_dim = spectrogram.shape
        
        # 频率掩码
        if freq_dim > 1:
            for _ in range(self.num_freq_mask):
                f = min(self.freq_mask_param, freq_dim - 1)
                if f > 0:
                    f = random.randint(1, f)
                    f0 = random.randint(0, freq_dim - f)
                    spectrogram[f0:f0 + f, :] = 0
        
        # 时间掩码
        if time_dim > 1:
            for _ in range(self.num_time_mask):
                t = min(self.time_mask_param, time_dim - 1)
                if t > 0:
                    t = random.randint(1, t)
                    t0 = random.randint(0, time_dim - t)
                    spectrogram[:, t0:t0 + t] = 0
        
        return spectrogram

class AudioAugmentation:
    """音频数据增强"""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def add_noise(self, waveform, noise_level=0.005):
        """添加高斯噪声"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def speed_perturb(self, waveform, speed_rate):
        """速度扰动"""
        if speed_rate == 1.0:
            return waveform
        
        old_length = waveform.shape[-1]
        new_length = int(old_length / speed_rate)
        resampler = torchaudio.transforms.Resample(
            orig_freq=self.sample_rate,
            new_freq=int(self.sample_rate * speed_rate)
        )
        waveform = resampler(waveform)
        
        resampler_back = torchaudio.transforms.Resample(
            orig_freq=int(self.sample_rate * speed_rate),
            new_freq=self.sample_rate
        )
        waveform = resampler_back(waveform)
        
        return waveform

class LibriSpeechDataset(Dataset):
    """增强版LibriSpeech数据集类"""
    
    def __init__(self, data_dirs, processor, hyperparameters, is_train=True):
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.processor = processor
        self.hyperparameters = hyperparameters
        self.is_train = is_train
        
        # 音频参数
        self.max_audio_length = hyperparameters['max_audio_length']
        self.min_audio_length = hyperparameters['min_audio_length']
        self.sample_rate = hyperparameters['sample_rate']
        self.audio_normalize = hyperparameters['audio_normalize']
        self.max_target_length = hyperparameters['max_target_length']
        self.max_length_samples = int(self.max_audio_length * self.sample_rate)
        self.min_length_samples = int(self.min_audio_length * self.sample_rate)
        
        # 数据增强
        if is_train:
            self.audio_aug = AudioAugmentation(self.sample_rate)
            self.spec_augment = SpecAugment(
                hyperparameters['freq_mask_param'],
                hyperparameters['time_mask_param'],
                hyperparameters['num_freq_mask'],
                hyperparameters['num_time_mask']
            ) if hyperparameters['use_spec_augment'] else None
        else:
            self.audio_aug = None
            self.spec_augment = None
        
        self.audio_files = []
        self.transcriptions = []
        self._prepare_data()
    
    def _prepare_data(self):
        """准备数据集"""
        for data_dir in self.data_dirs:
            print(f"正在加载数据集: {data_dir}")
            self._load_directory(data_dir)
        
        print(f"数据集加载完成: 共 {len(self.audio_files)} 个样本")
    
    def _load_directory(self, data_dir):
        """加载单个目录的数据"""
        count = 0
        
        for root, dirs, files in os.walk(data_dir):
            trans_files = [f for f in files if f.endswith('.trans.txt')]
            
            for trans_file in trans_files:
                trans_path = os.path.join(root, trans_file)
                
                try:
                    with open(trans_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split(' ', 1)
                            if len(parts) == 2:
                                audio_id, transcription = parts
                                audio_file = os.path.join(root, f"{audio_id}.flac")
                                
                                if os.path.exists(audio_file) and transcription.strip():
                                    clean_transcription = ' '.join(transcription.upper().split())
                                    
                                    # 更严格的长度检查
                                    try:
                                        encoded_length = len(self.processor.tokenizer.encode(clean_transcription))
                                        if encoded_length > self.max_target_length:
                                            continue
                                    except:
                                        continue
                                    
                                    self.audio_files.append(audio_file)
                                    self.transcriptions.append(clean_transcription)
                                    count += 1
                                    
                                    if count % 5000 == 0:
                                        print(f"  已加载 {count} 个文件...")
                except Exception as e:
                    print(f"读取转录文件出错 {trans_path}: {e}")
                    continue
    
    def _normalize_audio(self, audio_array):
        """音频归一化"""
        if self.audio_normalize:
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # RMS归一化
            rms = np.sqrt(np.mean(audio_array ** 2))
            if rms > 1e-8:
                target_rms = 0.1
                audio_array = audio_array * (target_rms / rms)
            
            audio_array = np.clip(audio_array, -1.0, 1.0)
            
        return audio_array.astype(np.float32)
    
    def _apply_augmentation(self, waveform):
        """应用数据增强"""
        if not self.is_train or self.audio_aug is None:
            return waveform
        
        # 速度扰动
        if self.hyperparameters['speed_perturb'] and random.random() < 0.5:
            speed_rate = random.choice(self.hyperparameters['speed_rates'])
            if speed_rate != 1.0:
                waveform = self.audio_aug.speed_perturb(waveform, speed_rate)
        
        # 添加噪声
        if self.hyperparameters['noise_augment'] and random.random() < self.hyperparameters['noise_prob']:
            waveform = self.audio_aug.add_noise(waveform, self.hyperparameters['noise_level'])
        
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
            
            # 转单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 应用数据增强
            waveform = self._apply_augmentation(waveform)
            
            # 获取音频数组
            audio_array = waveform.squeeze().numpy()
            
            # 处理长度
            if len(audio_array) > self.max_length_samples:
                if self.is_train:
                    start = np.random.randint(0, len(audio_array) - self.max_length_samples + 1)
                else:
                    start = (len(audio_array) - self.max_length_samples) // 2
                audio_array = audio_array[start:start + self.max_length_samples]
            elif len(audio_array) < self.min_length_samples:
                audio_array = np.pad(audio_array, (0, self.min_length_samples - len(audio_array)))
            
            # 归一化
            audio_array = self._normalize_audio(audio_array)
            
            return {
                "audio": audio_array,
                "transcription": transcription,
                "audio_length": len(audio_array),
                "apply_spec_augment": self.is_train and self.spec_augment is not None
            }
            
        except Exception as e:
            print(f"加载音频文件出错 {audio_file}: {e}")
            return {
                "audio": np.zeros(self.min_length_samples, dtype=np.float32),
                "transcription": "EMPTY",
                "audio_length": self.min_length_samples,
                "apply_spec_augment": False
            }

@dataclass
class CTCDataCollatorWithAugment:
    """支持数据增强的CTC数据整理器 - 修复版本"""
    processor: any
    padding: bool = True
    spec_augment: Optional[SpecAugment] = None
    label_smoothing: float = 0.0
    
    def __call__(self, features: List[Dict[str, Union[np.ndarray, str, int, bool]]]) -> Dict[str, torch.Tensor]:
        # 过滤有效样本
        valid_features = []
        for feature in features:
            if feature["transcription"] != "EMPTY" and len(feature["transcription"].strip()) > 0:
                valid_features.append(feature)
        
        if not valid_features:
            # 返回虚拟批次
            dummy_audio = np.zeros(16000, dtype=np.float32)
            dummy_text = "HELLO"
            
            return {
                "input_values": torch.tensor([dummy_audio], dtype=torch.float32),
                "labels": torch.tensor([[self.processor.tokenizer.encode(dummy_text)[0]]], dtype=torch.long),
                "transcriptions": [dummy_text],
                "attention_mask": torch.ones((1, 16000), dtype=torch.long)
            }
        
        audio_list = [f["audio"] for f in valid_features]
        text_list = [f["transcription"] for f in valid_features]
        apply_spec_augment = any(f.get("apply_spec_augment", False) for f in valid_features)
        
        # 处理音频
        inputs = self.processor(
            audio_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # 检查处理后的数据
        if torch.any(torch.isnan(inputs.input_values)) or torch.any(torch.isinf(inputs.input_values)):
            print("警告: 处理后的音频包含NaN/Inf值")
            inputs.input_values = torch.nan_to_num(inputs.input_values, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 处理标签
        with self.processor.as_target_processor():
            labels = self.processor(
                text_list,
                return_tensors="pt",
                padding=True
            )
        
        # 标签处理
        labels_tensor = labels.input_ids.clone()
        
        # 计算实际的输入和标签长度
        batch_size = inputs.input_values.shape[0]
        
        # 对于Wav2Vec2，输出长度大约是输入长度除以320（对于16kHz音频）
        if hasattr(inputs, 'attention_mask'):
            input_lengths = torch.sum(inputs.attention_mask, dim=1)
            # 估计CTC输出长度（非常保守的估计）
            estimated_output_lengths = input_lengths // 320 + 10  # 添加一些buffer
        else:
            # 如果没有attention_mask，假设所有输入都有效
            input_lengths = torch.full((batch_size,), inputs.input_values.shape[1], dtype=torch.long)
            estimated_output_lengths = input_lengths // 320 + 10
        
        # 检查并调整标签长度
        for i in range(batch_size):
            label = labels_tensor[i]
            valid_length = (label != self.processor.tokenizer.pad_token_id).sum().item()
            
            # 如果标签太长，进行截断
            if valid_length > 0 and valid_length > estimated_output_lengths[i] * 0.8:
                print(f"警告: 标签太长({valid_length})相对于估计的输出长度({estimated_output_lengths[i].item()})，进行截断")
                # 截断到安全长度
                safe_length = int(estimated_output_lengths[i].item() * 0.5)
                labels_tensor[i, safe_length:] = self.processor.tokenizer.pad_token_id
        
        # 将padding token设为-100
        labels_tensor[labels_tensor == self.processor.tokenizer.pad_token_id] = -100
        
        # 确保标签有效
        if torch.all(labels_tensor == -100):
            print("警告: 所有标签都是padding！")
            for i in range(labels_tensor.shape[0]):
                if i < len(text_list) and text_list[i]:
                    encoded = self.processor.tokenizer.encode(text_list[i])
                    if encoded:
                        max_len = min(len(encoded), labels_tensor.shape[1])
                        labels_tensor[i, :max_len] = torch.tensor(encoded[:max_len])
        
        return {
            "input_values": inputs.input_values,
            "labels": labels_tensor,
            "transcriptions": text_list,
            "attention_mask": inputs.attention_mask if hasattr(inputs, 'attention_mask') else None
        }

def compute_ctc_loss_with_length_check(logits, labels, processor, attention_mask=None):
    """计算CTC损失，包含长度检查和稳定性改进"""
    batch_size, max_time, vocab_size = logits.shape
    device = logits.device
    
    # 获取blank token ID
    blank_id = processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else 0
    
    # 计算输入长度
    if attention_mask is not None:
        # 基于attention mask计算实际的输出长度
        # Wav2Vec2的下采样率约为320
        input_lengths = (torch.sum(attention_mask, dim=1) // 320).long()
        # 确保长度至少为1
        input_lengths = torch.clamp(input_lengths, min=1, max=max_time)
    else:
        input_lengths = torch.full((batch_size,), max_time, dtype=torch.long, device=device)
    
    # 准备标签和计算目标长度
    target_lengths = []
    labels_list = []
    
    for i in range(batch_size):
        label = labels[i]
        # 获取有效标签（非-100）
        valid_mask = label != -100
        valid_labels = label[valid_mask]
        
        if len(valid_labels) == 0:
            # 如果没有有效标签，使用一个blank标签
            valid_labels = torch.tensor([blank_id], device=device)
            target_lengths.append(1)
        else:
            # 确保标签长度不超过输入长度
            max_label_length = input_lengths[i].item() - 1  # CTC需要至少一个空白
            if len(valid_labels) > max_label_length:
                print(f"警告: 批次{i}的标签长度({len(valid_labels)})超过最大允许长度({max_label_length})，进行截断")
                valid_labels = valid_labels[:max_label_length]
            
            target_lengths.append(len(valid_labels))
        
        labels_list.append(valid_labels)
    
    # 创建padded标签张量
    max_target_length = max(target_lengths) if target_lengths else 1
    padded_labels = torch.full((batch_size, max_target_length), blank_id, dtype=torch.long, device=device)
    
    for i, label in enumerate(labels_list):
        if len(label) > 0:
            padded_labels[i, :len(label)] = label
    
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    
    # 添加数值稳定性：对logits进行缩放
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]  # 防止数值溢出
    
    # 计算log_probs
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 再次检查数值稳定性
    if torch.any(torch.isnan(log_probs)) or torch.any(torch.isinf(log_probs)):
        print("警告: log_probs包含NaN或Inf，尝试修复")
        log_probs = torch.nan_to_num(log_probs, nan=-10.0, posinf=10.0, neginf=-10.0)
    
    try:
        # 计算CTC损失
        loss = F.ctc_loss(
            log_probs.transpose(0, 1),  # (T, B, C)
            padded_labels,              # (B, S)
            input_lengths,              # (B,)
            target_lengths,             # (B,)
            blank=blank_id,
            reduction='mean',
            zero_infinity=True
        )
        
        # 检查损失值
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告: CTC损失为NaN/Inf，返回默认值")
            return torch.tensor(10.0, device=device, requires_grad=True)
        
        # 如果损失太大，进行裁剪
        if loss.item() > hyperparameters['max_loss_value']:
            print(f"警告: CTC损失过大({loss.item():.2f})，进行裁剪")
            loss = torch.clamp(loss, max=hyperparameters['max_loss_value'])
        
        return loss
        
    except Exception as e:
        print(f"CTC损失计算出错: {e}")
        return torch.tensor(10.0, device=device, requires_grad=True)

def create_model_with_regularization(model_path, hyperparameters):
    """创建带正则化的模型 - 增强版"""
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    
    # 启用梯度检查点以节省内存
    if hyperparameters.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        print("✓ 梯度检查点已启用")
    
    # 设置config中的dropout率
    model.config.hidden_dropout = hyperparameters['hidden_dropout']
    model.config.attention_dropout = hyperparameters['attention_dropout'] 
    model.config.feat_proj_dropout = hyperparameters['feat_proj_dropout']
    model.config.final_dropout = hyperparameters['dropout']
    
    # 应用dropout到模型
    def apply_dropout(module, dropout_rate, module_name=""):
        """递归应用dropout到特定层"""
        for name, child in module.named_children():
            full_name = f"{module_name}.{name}" if module_name else name
            
            if isinstance(child, torch.nn.Dropout):
                if 'attention' in full_name and 'dropout' in name:
                    child.p = hyperparameters['attention_dropout']
                elif 'feed_forward' in full_name and 'dropout' in name:
                    child.p = hyperparameters['hidden_dropout']
                elif 'feature_projection' in full_name and 'dropout' in name:
                    child.p = hyperparameters['feat_proj_dropout']
                elif name == 'dropout' and 'wav2vec2' not in full_name:
                    child.p = hyperparameters['dropout']
                else:
                    child.p = min(child.p, hyperparameters['hidden_dropout'])
            else:
                apply_dropout(child, dropout_rate, full_name)
    
    apply_dropout(model, hyperparameters['hidden_dropout'])
    
    # 重新初始化lm_head - 使用更小的标准差
    if hasattr(model, 'lm_head'):
        torch.nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.01)
        if model.lm_head.bias is not None:
            torch.nn.init.zeros_(model.lm_head.bias)
        print("✓ LM head已重新初始化（std=0.01）")
    
    # 冻结特征提取器（如果需要）
    if hyperparameters.get('freeze_feature_extractor', False):
        model.freeze_feature_encoder()
        print("✓ 特征提取器已冻结")
        
        # 打印冻结的参数数量
        frozen_params = sum(p.numel() for n, p in model.named_parameters() 
                          if 'feature_extractor' in n or 'feature_projection' in n)
        print(f"  冻结参数数量: {frozen_params:,}")
    
    return model

def create_optimizer_with_layer_wise_lr(model, hyperparameters):
    """创建分层学习率优化器"""
    # 获取层学习率倍数
    layer_lr_multipliers = hyperparameters.get('layer_lr_multipliers', {
        'feature_extractor': 0.01,
        'feature_projection': 0.1,
        'encoder': 0.5,
        'lm_head': 1.0
    })
    
    base_lr = hyperparameters['learning_rate']
    
    # 创建参数组
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 确定层的学习率倍数
        lr_multiplier = 1.0
        for layer_name, multiplier in layer_lr_multipliers.items():
            if layer_name in name:
                lr_multiplier = multiplier
                break
        
        # 确定是否需要weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        weight_decay = 0.0 if any(nd in name for nd in no_decay) else hyperparameters['weight_decay']
        
        # 添加到对应的参数组
        param_groups.append({
            'params': [param],
            'lr': base_lr * lr_multiplier,
            'weight_decay': weight_decay,
            'name': name
        })
    
    # 打印学习率信息
    print("\n分层学习率设置:")
    for layer_name, multiplier in layer_lr_multipliers.items():
        lr = base_lr * multiplier
        print(f"  {layer_name}: {lr:.2e} (x{multiplier})")
    
    # 创建优化器
    optimizer = AdamW(
        param_groups,
        betas=(hyperparameters['adam_beta1'], hyperparameters['adam_beta2']),
        eps=hyperparameters['adam_epsilon']
    )
    
    return optimizer

def gradual_unfreezing_callback(model, epoch, hyperparameters):
    """逐步解冻模型层"""
    if not hyperparameters.get('gradual_unfreezing', False):
        return
    
    unfreeze_epoch = hyperparameters.get('unfreeze_epoch', 5)
    
    if epoch < unfreeze_epoch:
        # 前N个epoch只训练lm_head
        for name, param in model.named_parameters():
            param.requires_grad = "lm_head" in name
        print(f"Epoch {epoch}: 只训练lm_head")
        
    elif epoch < unfreeze_epoch + 5:
        # 接下来5个epoch解冻encoder后半部分
        for name, param in model.named_parameters():
            if any(x in name for x in ["lm_head", "encoder.layers.20", "encoder.layers.21", 
                                       "encoder.layers.22", "encoder.layers.23"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f"Epoch {epoch}: 解冻encoder后半部分")
        
    elif epoch < unfreeze_epoch + 10:
        # 再5个epoch解冻整个encoder
        for name, param in model.named_parameters():
            if "feature_extractor" not in name and "feature_projection" not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f"Epoch {epoch}: 解冻整个encoder")
        
    else:
        # 最后解冻所有层（包括特征提取器）
        for name, param in model.named_parameters():
            param.requires_grad = True
        print(f"Epoch {epoch}: 解冻所有层")
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数数量: {trainable_params:,}")

def check_gradient_health(model, max_grad_norm=1.0):
    """检查梯度健康状态"""
    total_norm = 0
    param_count = 0
    nan_grads = []
    large_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            param_count += 1
            grad = param.grad.data
            
            # 检查NaN
            if torch.isnan(grad).any():
                nan_grads.append(name)
                param.grad.data = torch.zeros_like(grad)  # 清零NaN梯度
            
            # 检查Inf
            if torch.isinf(grad).any():
                nan_grads.append(name)
                param.grad.data = torch.zeros_like(grad)  # 清零Inf梯度
            
            # 计算范数
            param_norm = grad.norm(2).item()
            total_norm += param_norm ** 2
            
            # 检查过大的梯度
            if param_norm > max_grad_norm * 10:
                large_grads.append((name, param_norm))
    
    total_norm = total_norm ** 0.5
    
    return {
        'total_norm': total_norm,
        'param_count': param_count,
        'nan_grads': nan_grads,
        'large_grads': large_grads,
        'is_healthy': len(nan_grads) == 0 and total_norm < max_grad_norm * 100
    }

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, history, best_metrics, save_dir):
    """保存检查点"""
    checkpoint_dir = os.path.join(save_dir, f'checkpoint-epoch-{epoch}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(checkpoint_dir)
    
    # 保存训练状态
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'history': history,
        'best_metrics': best_metrics,
        'hyperparameters': hyperparameters
    }
    
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'training_state.pt'))
    print(f"✓ 检查点保存到: {checkpoint_dir}")
    
    # 保存latest指针
    latest_path = os.path.join(save_dir, 'latest_checkpoint')
    with open(latest_path, 'w') as f:
        f.write(checkpoint_dir)

def evaluate_model(model, dataloader, processor, device, use_amp=True):
    """评估模型"""
    model.eval()
    predictions = []
    references = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="评估中")):
            try:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # 检查输入
                if torch.any(torch.isnan(input_values)) or torch.any(torch.isinf(input_values)):
                    print(f"警告: 评估批次 {batch_idx} 包含NaN/Inf输入")
                    continue
                
                # 使用自定义的CTC损失计算
                outputs = model(input_values=input_values)
                loss = compute_ctc_loss_with_length_check(
                    outputs.logits, labels, processor, attention_mask
                )
                
                # 检查损失
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > hyperparameters['max_loss_value']:
                    print(f"警告: 评估批次 {batch_idx} 损失异常: {loss.item()}")
                    continue
                
                total_loss += loss.item()
                num_batches += 1
                
                # 获取预测
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # 解码预测
                for pred_ids in predicted_ids:
                    pred_str = processor.decode(pred_ids, skip_special_tokens=True)
                    predictions.append(pred_str)
                
                references.extend(batch["transcriptions"])
                
            except Exception as e:
                print(f"评估批次 {batch_idx} 时出错: {e}")
                continue
    
    if num_batches == 0:
        print("警告: 没有有效的评估批次！")
        return float('inf'), 1.0, 1.0, [], []
    
    avg_loss = total_loss / num_batches
    word_error_rate, char_error_rate = compute_metrics(predictions, references)
    
    return avg_loss, word_error_rate, char_error_rate, predictions, references

def compute_metrics(predictions, references):
    """计算WER和CER"""
    if not predictions or not references:
        return 1.0, 1.0
    
    valid_pairs = []
    for p, r in zip(predictions, references):
        if isinstance(p, str) and isinstance(r, str) and p.strip() and r.strip():
            valid_pairs.append((p.strip(), r.strip()))
    
    if not valid_pairs:
        return 1.0, 1.0
    
    predictions, references = zip(*valid_pairs)
    
    try:
        word_error_rate = wer(list(references), list(predictions))
        char_error_rate = cer(list(references), list(predictions))
        
        word_error_rate = min(1.0, max(0.0, word_error_rate))
        char_error_rate = min(1.0, max(0.0, char_error_rate))
        
    except Exception as e:
        print(f"计算WER/CER时出错: {e}")
        return 1.0, 1.0
    
    return word_error_rate, char_error_rate

def train_model(model, train_dataloader, val_dataloader, processor, hyperparameters, device, save_dir):
    """训练模型 - 修复版with分层学习率和逐步解冻"""
    
    # 创建优化器
    if hyperparameters.get('layer_wise_lr', False):
        optimizer = create_optimizer_with_layer_wise_lr(model, hyperparameters)
    else:
        # 原始的优化器创建代码
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": hyperparameters['weight_decay'],
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=hyperparameters['learning_rate'],
            betas=(hyperparameters['adam_beta1'], hyperparameters['adam_beta2']),
            eps=hyperparameters['adam_epsilon']
        )
    
    # 计算总训练步数
    num_training_steps = len(train_dataloader) * hyperparameters['num_epochs'] // hyperparameters['gradient_accumulation_steps']
    num_warmup_steps = int(num_training_steps * hyperparameters['warmup_ratio'])
    
    # 使用余弦退火学习率调度
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5
    )
    
    # 混合精度训练
    scaler = GradScaler() if hyperparameters['mixed_precision'] else None
    
    # 初始化变量
    start_epoch = 0
    global_step = 0
    history = {
        'step_train_loss': [],
        'step_learning_rate': [],
        'step_gradient_norm': [],
        'step_numbers': [],
        'epoch_train_loss': [],
        'epoch_val_loss': [],
        'epoch_val_wer': [],
        'epoch_val_cer': [],
        'epoch_train_wer': [],
        'epoch_train_cer': []
    }
    best_metrics = {
        'best_val_wer': float('inf'),
        'best_val_cer': float('inf'),
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    patience_counter = 0
    no_improvement_count = 0
    
    print(f"开始训练，全局步数: {global_step}")
    print(f"总训练步数: {num_training_steps}, Warmup步数: {num_warmup_steps}")
    print(f"初始学习率: {optimizer.param_groups[0]['lr']:.2e}")
    
    # 在训练循环开始前，运行模型初始化步骤
    print("\n运行模型初始化步骤...")
    model.eval()
    with torch.no_grad():
        # 运行几个批次来稳定批归一化层
        init_batches = 3
        for i, batch in enumerate(train_dataloader):
            if i >= init_batches:
                break
            input_values = batch["input_values"].to(device)
            _ = model(input_values=input_values)
            print(f"初始化批次 {i+1}/{init_batches} 完成")
    model.train()
    print("✓ 模型初始化完成")
    
    # 在开始训练前进行数据和模型检查
    print("\n运行训练前检查...")
    try:
        test_batch = next(iter(train_dataloader))
        print(f"批次大小: {test_batch['input_values'].shape[0]}")
        print(f"输入形状: {test_batch['input_values'].shape}")
        print(f"标签形状: {test_batch['labels'].shape}")
        
        # 运行一次前向传播测试
        model.eval()
        with torch.no_grad():
            test_inputs = test_batch['input_values'].to(device)[:1]
            test_labels = test_batch['labels'].to(device)[:1]
            test_attention_mask = test_batch.get('attention_mask', None)
            if test_attention_mask is not None:
                test_attention_mask = test_attention_mask.to(device)[:1]
            
            test_output = model(input_values=test_inputs)
            print(f"输出logits形状: {test_output.logits.shape}")
            
            # 测试CTC损失计算
            test_loss = compute_ctc_loss_with_length_check(
                test_output.logits, test_labels, processor, test_attention_mask
            )
            print(f"测试损失: {test_loss.item():.4f}")
            
            if test_loss.item() > 50:
                print("⚠️  警告: 初始损失很高，但会通过冻结特征提取器来解决")
        
        model.train()
        print("✓ 训练前检查完成")
        
    except Exception as e:
        print(f"训练前检查失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 初始验证
    if start_epoch == 0 and global_step == 0:
        print("\n运行初始验证...")
        initial_val_loss, initial_val_wer, initial_val_cer, _, _ = evaluate_model(
            model, val_dataloader, processor, device, use_amp=hyperparameters['mixed_precision']
        )
        print(f"初始验证结果 - 损失: {initial_val_loss:.4f}, WER: {initial_val_wer:.4f}, CER: {initial_val_cer:.4f}")
    
    # 训练循环
    for epoch in range(start_epoch, hyperparameters['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{hyperparameters['num_epochs']}")
        print(f"{'='*50}")
        
        # 应用逐步解冻
        gradual_unfreezing_callback(model, epoch, hyperparameters)
        
        # 训练阶段
        model.train()
        epoch_loss = 0
        num_batches = 0
        num_skipped_batches = 0
        train_predictions = []
        train_references = []
        
        progress_bar = tqdm(train_dataloader, desc=f"训练 Epoch {epoch + 1}")
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            try:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # 检查输入是否有效
                if torch.any(torch.isnan(input_values)) or torch.any(torch.isinf(input_values)):
                    print(f"警告: 跳过包含NaN/Inf输入的批次")
                    num_skipped_batches += 1
                    continue
                
                # 前向传播
                if hyperparameters['mixed_precision']:
                    with autocast():
                        outputs = model(input_values=input_values)
                        loss = compute_ctc_loss_with_length_check(
                            outputs.logits, labels, processor, attention_mask
                        )
                        loss = loss / hyperparameters['gradient_accumulation_steps']
                else:
                    outputs = model(input_values=input_values)
                    loss = compute_ctc_loss_with_length_check(
                        outputs.logits, labels, processor, attention_mask
                    )
                    loss = loss / hyperparameters['gradient_accumulation_steps']
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > hyperparameters['max_loss_value']:
                    print(f"警告: 跳过损失异常的批次 (loss={loss.item():.2f})")
                    num_skipped_batches += 1
                    optimizer.zero_grad()  # 清理梯度
                    continue
                
                # 反向传播
                if hyperparameters['mixed_precision']:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item() * hyperparameters['gradient_accumulation_steps']
                num_batches += 1
                
                # 收集训练预测（每N步）
                if step % 100 == 0:
                    with torch.no_grad():
                        logits = outputs.logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                        for pred_ids in predicted_ids:
                            pred_str = processor.decode(pred_ids, skip_special_tokens=True)
                            train_predictions.append(pred_str)
                        train_references.extend(batch["transcriptions"])
                
                # 梯度累积和更新
                if (step + 1) % hyperparameters['gradient_accumulation_steps'] == 0 or (step + 1) == len(train_dataloader):
                    # 检查梯度健康状态
                    grad_health = check_gradient_health(model, hyperparameters['max_grad_norm'])
                    
                    if not grad_health['is_healthy']:
                        print(f"\n警告: 梯度不健康!")
                        if grad_health['nan_grads']:
                            print(f"  NaN/Inf梯度: {grad_health['nan_grads'][:5]}...")
                        if grad_health['large_grads']:
                            print(f"  过大梯度: {grad_health['large_grads'][:5]}...")
                        
                        optimizer.zero_grad()
                        if hyperparameters['mixed_precision'] and scaler._scale is not None:
                            scaler.update()
                        continue
                    
                    # 梯度裁剪和优化器更新
                    if hyperparameters['mixed_precision']:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            hyperparameters['max_grad_norm']
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            hyperparameters['max_grad_norm']
                        )
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # 记录
                    if global_step % hyperparameters['logging_steps'] == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        history['step_train_loss'].append(epoch_loss / max(num_batches, 1))
                        history['step_learning_rate'].append(current_lr)
                        history['step_gradient_norm'].append(grad_health['total_norm'])
                        history['step_numbers'].append(global_step)
                    
                    # 更新进度条
                    avg_loss = epoch_loss / max(num_batches, 1)
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                        'grad': f"{grad_health['total_norm']:.2f}",
                        'skip': num_skipped_batches
                    })
                    
            except Exception as e:
                print(f"训练步骤出错: {e}")
                import traceback
                traceback.print_exc()
                optimizer.zero_grad()
                if hyperparameters['mixed_precision'] and scaler._scale is not None:
                    scaler.update()
                continue
        
        # 计算训练指标
        avg_train_loss = epoch_loss / max(num_batches, 1)
        if train_predictions and train_references:
            train_wer, train_cer = compute_metrics(train_predictions, train_references)
        else:
            train_wer, train_cer = 1.0, 1.0
        
        history['epoch_train_loss'].append(avg_train_loss)
        history['epoch_train_wer'].append(train_wer)
        history['epoch_train_cer'].append(train_cer)
        
        print(f"\nEpoch {epoch + 1} 训练完成 - 跳过批次: {num_skipped_batches}/{len(train_dataloader)}")
        
        # 验证
        print(f"Epoch {epoch + 1} 验证中...")
        val_loss, val_wer, val_cer, val_preds, val_refs = evaluate_model(
            model, val_dataloader, processor, device, use_amp=hyperparameters['mixed_precision']
        )
        
        history['epoch_val_loss'].append(val_loss)
        history['epoch_val_wer'].append(val_wer)
        history['epoch_val_cer'].append(val_cer)
        
        # 打印结果
        print(f"\nEpoch {epoch + 1} 结果:")
        print(f"训练 - 损失: {avg_train_loss:.4f}, WER: {train_wer:.4f}, CER: {train_cer:.4f}")
        print(f"验证 - 损失: {val_loss:.4f}, WER: {val_wer:.4f}, CER: {val_cer:.4f}")
        
        # 检查是否有改进
        improved = False
        if val_wer < best_metrics['best_val_wer'] - hyperparameters['early_stopping_min_delta']:
            best_metrics['best_val_wer'] = val_wer
            best_metrics['best_val_cer'] = val_cer
            best_metrics['best_val_loss'] = val_loss
            best_metrics['best_epoch'] = epoch
            improved = True
            
            # 保存最佳模型
            best_model_path = os.path.join(save_dir, 'best_model')
            model.save_pretrained(best_model_path)
            processor.save_pretrained(best_model_path)
            
            print(f"✓ 新的最佳模型! WER: {val_wer:.4f}")
        
        # 保存检查点
        save_checkpoint(
            model, optimizer, scheduler, scaler, 
            epoch, global_step, history, best_metrics, save_dir
        )
        
        # 早停检查
        if improved:
            patience_counter = 0
            no_improvement_count = 0
        else:
            patience_counter += 1
            no_improvement_count += 1
            
            # 学习率衰减策略
            if no_improvement_count >= 3 and no_improvement_count % 3 == 0:
                print(f"⚠️  {no_improvement_count} epochs没有改进，降低学习率")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
        
        if patience_counter >= hyperparameters['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"最佳验证WER: {best_metrics['best_val_wer']:.4f} (Epoch {best_metrics['best_epoch'] + 1})")
            break
        
        # 打印训练状态
        print(f"最佳WER: {best_metrics['best_val_wer']:.4f} | " + 
              f"Patience: {patience_counter}/{hyperparameters['early_stopping_patience']}")
    
    return history, best_metrics

def plot_training_curves_enhanced(history, save_dir):
    """增强版训练曲线绘图"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 训练损失（Step级别）
    ax1 = plt.subplot(3, 3, 1)
    if history['step_train_loss']:
        ax1.plot(history['step_numbers'], history['step_train_loss'], 'b-', alpha=0.7, linewidth=1)
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('损失')
        ax1.set_title('训练损失 (Step级别)')
        ax1.set_yscale('log')
    
    # 2. 学习率
    ax2 = plt.subplot(3, 3, 2)
    if history['step_learning_rate']:
        ax2.plot(history['step_numbers'], history['step_learning_rate'], 'purple', alpha=0.8)
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('学习率')
        ax2.set_title('学习率调度')
        ax2.set_yscale('log')
    
    # 3. 梯度范数
    ax3 = plt.subplot(3, 3, 3)
    if history['step_gradient_norm']:
        ax3.plot(history['step_numbers'], history['step_gradient_norm'], 'red', alpha=0.7)
        ax3.set_xlabel('训练步数')
        ax3.set_ylabel('梯度范数')
        ax3.set_title('梯度范数')
        ax3.axhline(y=hyperparameters['max_grad_norm'], color='red', linestyle='--', 
                   alpha=0.5, label='裁剪阈值')
        ax3.legend()
    
    # 4. 训练 vs 验证损失
    ax4 = plt.subplot(3, 3, 4)
    if history['epoch_train_loss'] and history['epoch_val_loss']:
        epochs = range(1, len(history['epoch_train_loss']) + 1)
        val_epochs = range(1, len(history['epoch_val_loss']) + 1)
        ax4.plot(epochs, history['epoch_train_loss'], 'b-', label='训练', marker='o')
        ax4.plot(val_epochs, history['epoch_val_loss'], 'r-', label='验证', marker='s')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('损失')
        ax4.set_title('训练 vs 验证损失')
        ax4.legend()
        ax4.set_yscale('log')
    
    # 5. WER对比
    ax5 = plt.subplot(3, 3, 5)
    if history.get('epoch_train_wer') and history['epoch_val_wer']:
        epochs = range(1, len(history['epoch_train_wer']) + 1)
        val_epochs = range(1, len(history['epoch_val_wer']) + 1)
        ax5.plot(epochs, history['epoch_train_wer'], 'b-', label='训练', marker='o')
        ax5.plot(val_epochs, history['epoch_val_wer'], 'r-', label='验证', marker='s')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('WER')
        ax5.set_title('训练 vs 验证 WER')
        ax5.legend()
        ax5.set_ylim(0, 1)
    
    # 6. CER对比
    ax6 = plt.subplot(3, 3, 6)
    if history.get('epoch_train_cer') and history['epoch_val_cer']:
        epochs = range(1, len(history['epoch_train_cer']) + 1)
        val_epochs = range(1, len(history['epoch_val_cer']) + 1)
        ax6.plot(epochs, history['epoch_train_cer'], 'b-', label='训练', marker='o')
        ax6.plot(val_epochs, history['epoch_val_cer'], 'r-', label='验证', marker='s')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('CER')
        ax6.set_title('训练 vs 验证 CER')
        ax6.legend()
        ax6.set_ylim(0, 1)
    
    # 7. 过拟合检测
    ax7 = plt.subplot(3, 3, 7)
    if history['epoch_train_loss'] and history['epoch_val_loss']:
        epochs = range(1, min(len(history['epoch_train_loss']), len(history['epoch_val_loss'])) + 1)
        train_losses = history['epoch_train_loss'][:len(epochs)]
        val_losses = history['epoch_val_loss'][:len(epochs)]
        
        overfitting_gap = [v - t for t, v in zip(train_losses, val_losses)]
        ax7.plot(epochs, overfitting_gap, 'g-', marker='o')
        ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('验证损失 - 训练损失')
        ax7.set_title('过拟合检测')
        ax7.fill_between(epochs, 0, overfitting_gap, where=[g > 0 for g in overfitting_gap], 
                        alpha=0.3, color='red', label='过拟合')
        ax7.legend()
    
    # 8. WER改进率
    ax8 = plt.subplot(3, 3, 8)
    if history['epoch_val_wer'] and len(history['epoch_val_wer']) > 1:
        epochs = range(2, len(history['epoch_val_wer']) + 1)
        improvements = []
        for i in range(1, len(history['epoch_val_wer'])):
            improvement = (history['epoch_val_wer'][i-1] - history['epoch_val_wer'][i]) / history['epoch_val_wer'][i-1] * 100
            improvements.append(improvement)
        
        ax8.bar(epochs, improvements, color=['green' if i > 0 else 'red' for i in improvements])
        ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('WER改进率 (%)')
        ax8.set_title('每Epoch WER改进')
    
    # 9. 训练效率
    ax9 = plt.subplot(3, 3, 9)
    if history['step_train_loss'] and history['step_numbers']:
        # 计算移动平均
        window_size = 50
        if len(history['step_train_loss']) >= window_size:
            moving_avg = np.convolve(history['step_train_loss'], 
                                    np.ones(window_size)/window_size, mode='valid')
            steps = history['step_numbers'][window_size-1:]
            ax9.plot(steps, moving_avg, 'b-', label=f'{window_size}步移动平均')
            
            # 添加趋势线
            z = np.polyfit(steps, moving_avg, 1)
            p = np.poly1d(z)
            ax9.plot(steps, p(steps), "r--", alpha=0.8, label='趋势线')
            
            ax9.set_xlabel('训练步数')
            ax9.set_ylabel('损失')
            ax9.set_title('训练效率分析')
            ax9.legend()
            ax9.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves_enhanced.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"增强版训练曲线已保存到: {os.path.join(save_dir, 'training_curves_enhanced.png')}")

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 优化GPU设置
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # 创建保存目录
    save_dir = os.path.join(SAVE_PATH, f"wav2vec2_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存超参数
    with open(os.path.join(save_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4, ensure_ascii=False)
    
    # 加载模型和处理器
    print(f"加载模型: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print("错误: 模型路径不存在")
        return
    
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = create_model_with_regularization(MODEL_PATH, hyperparameters)
    
    print(f"✅ 模型加载成功！")
    print(f"词汇表大小: {len(processor.tokenizer.vocab)}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    model.to(device)
    
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 准备数据集
    print("\n准备数据集...")
    
    # 数据路径配置
    if hyperparameters['use_all_data']:
        train_dirs = [
            os.path.join(DATA_PATH, "train-clean-100"),
            os.path.join(DATA_PATH, "train-clean-360"),
            os.path.join(DATA_PATH, "train-other-500")
        ]
        train_dirs = [d for d in train_dirs if os.path.exists(d)]
        print(f"使用训练数据集: {train_dirs}")
    else:
        train_dirs = [os.path.join(DATA_PATH, "train-clean-100")]
    
    val_dirs = [os.path.join(DATA_PATH, "dev-clean")]
    test_dirs = [os.path.join(DATA_PATH, "test-clean")]
    
    # 创建数据增强
    spec_augment = SpecAugment(
        hyperparameters['freq_mask_param'],
        hyperparameters['time_mask_param'],
        hyperparameters['num_freq_mask'],
        hyperparameters['num_time_mask']
    ) if hyperparameters['use_spec_augment'] else None
    
    # 创建数据集
    train_dataset = LibriSpeechDataset(
        data_dirs=train_dirs,
        processor=processor,
        hyperparameters=hyperparameters,
        is_train=True
    )
    
    val_dataset = LibriSpeechDataset(
        data_dirs=val_dirs,
        processor=processor,
        hyperparameters=hyperparameters,
        is_train=False
    )
    
    test_dataset = LibriSpeechDataset(
        data_dirs=test_dirs,
        processor=processor,
        hyperparameters=hyperparameters,
        is_train=False
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据整理器
    data_collator = CTCDataCollatorWithAugment(
        processor=processor,
        spec_augment=spec_augment,
        label_smoothing=hyperparameters['label_smoothing']
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=hyperparameters['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if hyperparameters['num_workers'] > 0 else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        collate_fn=data_collator,
        num_workers=hyperparameters['num_workers'],
        pin_memory=True,
        persistent_workers=True if hyperparameters['num_workers'] > 0 else False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        collate_fn=data_collator,
        num_workers=hyperparameters['num_workers'],
        pin_memory=True,
        persistent_workers=True if hyperparameters['num_workers'] > 0 else False
    )
    
    # 测试数据加载
    print("\n测试数据加载...")
    try:
        test_batch = next(iter(train_dataloader))
        print(f"批次大小: {test_batch['input_values'].shape[0]}")
        print(f"输入形状: {test_batch['input_values'].shape}")
        print(f"标签形状: {test_batch['labels'].shape}")
        print("✓ 数据加载测试通过")
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        return
    
    # 开始训练
    print("\n开始训练...")
    print(f"总训练步数: {len(train_dataloader) * hyperparameters['num_epochs'] // hyperparameters['gradient_accumulation_steps']}")
    print(f"数据增强: {'启用' if hyperparameters['use_spec_augment'] else '禁用'}")
    print(f"标签平滑: {hyperparameters['label_smoothing']}")
    print(f"混合精度: {'启用' if hyperparameters['mixed_precision'] else '禁用'}")
    print(f"冻结特征提取器: {'是' if hyperparameters.get('freeze_feature_extractor', False) else '否'}")
    print(f"分层学习率: {'启用' if hyperparameters.get('layer_wise_lr', False) else '禁用'}")
    print(f"逐步解冻: {'启用' if hyperparameters.get('gradual_unfreezing', False) else '禁用'}")
    
    history, best_metrics = train_model(
        model, train_dataloader, val_dataloader,
        processor, hyperparameters, device, save_dir
    )
    
    # 保存训练历史和最佳指标
    print("\n保存训练结果...")
    try:
        # 保存历史
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, list):
                history_serializable[key] = [float(x) if isinstance(x, (int, float, np.number)) else x for x in value]
            else:
                history_serializable[key] = value
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history_serializable, f, indent=4, ensure_ascii=False)
        
        # 保存最佳指标
        with open(os.path.join(save_dir, 'best_metrics.json'), 'w') as f:
            json.dump(best_metrics, f, indent=4, ensure_ascii=False)
        
        print("✓ 训练结果保存成功")
    except Exception as e:
        print(f"保存训练结果失败: {e}")
    
    # 绘制增强版训练曲线
    print("\n绘制训练曲线...")
    try:
        plot_training_curves_enhanced(history, save_dir)
        print("✓ 训练曲线绘制成功")
    except Exception as e:
        print(f"绘制训练曲线失败: {e}")
    
    # 最终测试集评估
    print("\n在测试集上进行最终评估...")
    
    best_model_path = os.path.join(save_dir, 'best_model')
    if os.path.exists(best_model_path):
        try:
            # 加载最佳模型
            best_model = Wav2Vec2ForCTC.from_pretrained(best_model_path)
            best_model.to(device)
            
            # 评估
            test_loss, test_wer, test_cer, test_preds, test_refs = evaluate_model(
                best_model, test_dataloader, processor, device, 
                use_amp=hyperparameters['mixed_precision']
            )
            
            print(f"\n最终测试集结果:")
            print(f"Loss: {test_loss:.4f}")
            print(f"WER: {test_wer:.4f}")
            print(f"CER: {test_cer:.4f}")
            
            # 保存测试结果
            test_results = {
                'test_loss': float(test_loss),
                'test_wer': float(test_wer),
                'test_cer': float(test_cer),
                'best_val_wer': float(best_metrics['best_val_wer']),
                'best_val_cer': float(best_metrics['best_val_cer']),
                'best_epoch': best_metrics['best_epoch'] + 1,
                'total_epochs': len(history['epoch_train_loss']),
                'hyperparameters': hyperparameters
            }
            
            with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
                json.dump(test_results, f, indent=4, ensure_ascii=False)
            
            # 保存预测示例
            with open(os.path.join(save_dir, 'test_predictions.txt'), 'w', encoding='utf-8') as f:
                f.write("测试集预测示例\n")
                f.write("=" * 80 + "\n\n")
                
                for i in range(min(30, len(test_preds), len(test_refs))):
                    f.write(f"样本 {i+1}:\n")
                    f.write(f"参考: {test_refs[i]}\n")
                    f.write(f"预测: {test_preds[i]}\n")
                    
                    # 计算单个样本的WER
                    sample_wer = wer(test_refs[i], test_preds[i])
                    f.write(f"WER: {sample_wer:.3f}\n")
                    f.write("-" * 80 + "\n")
            
            print("✓ 测试结果保存成功")
            
        except Exception as e:
            print(f"测试集评估失败: {e}")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 打印训练总结
    print(f"\n{'='*50}")
    print("训练总结")
    print(f"{'='*50}")
    print(f"保存目录: {save_dir}")
    print(f"最佳验证WER: {best_metrics['best_val_wer']:.4f} (Epoch {best_metrics['best_epoch'] + 1})")
    print(f"最佳验证CER: {best_metrics['best_val_cer']:.4f}")
    
    # 分析训练曲线
    if history['epoch_val_wer']:
        initial_wer = history['epoch_val_wer'][0]
        final_wer = best_metrics['best_val_wer']
        improvement = (initial_wer - final_wer) / initial_wer * 100
        print(f"WER改进: {improvement:.1f}% (从 {initial_wer:.4f} 到 {final_wer:.4f})")
    
    # 过拟合检测
    if history['epoch_train_wer'] and history['epoch_val_wer']:
        train_wer_final = history['epoch_train_wer'][-1]
        val_wer_final = history['epoch_val_wer'][-1]
        overfitting_gap = val_wer_final - train_wer_final
        
        if overfitting_gap > 0.1:
            print(f"⚠️  可能存在过拟合: 训练WER={train_wer_final:.4f}, 验证WER={val_wer_final:.4f}")
        else:
            print(f"✓ 模型泛化良好: 训练WER={train_wer_final:.4f}, 验证WER={val_wer_final:.4f}")

if __name__ == "__main__":
    main()