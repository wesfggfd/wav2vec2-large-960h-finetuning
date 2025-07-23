#!/usr/bin/env python3
"""
Wav2Vec2 微调脚本 - 基于最佳模型进行改进
完整实现版本
"""

import os
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
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
SAVE_PATH = os.path.join(BASE_PATH, ".cache")

# 微调超参数配置
finetune_hyperparameters = {
    # 基础训练参数 - 针对微调优化
    'learning_rate': 5e-6,              # 更小的初始学习率
    'min_learning_rate': 1e-8,      
    'batch_size': 8,                    # 更小的批次
    'num_epochs': 20,                   
    'weight_decay': 0.01,               # 增加权重衰减
    'max_grad_norm': 0.5,               # 更严格的梯度裁剪
    'warmup_ratio': 0.1,                
    'gradient_accumulation_steps': 8,   # 增加梯度累积
    
    # 学习率调度策略
    'lr_scheduler': 'cosine_warm_restarts',
    'T_0': 5,                           # 第一个周期的epoch数
    'T_mult': 2,                        # 周期增长因子
    'eta_min': 1e-8,                    
    
    # 正则化策略
    'dropout': 0.15,                    # 增加dropout
    'attention_dropout': 0.15,    
    'feat_proj_dropout': 0.15,
    'hidden_dropout': 0.15,
    'label_smoothing': 0.1,             # 添加标签平滑
    
    # 层解冻策略
    'freeze_feature_extractor': True,
    'gradual_unfreezing': True,
    'unfreeze_schedule': {
        0: ['lm_head'],
        3: ['encoder.layers.20', 'encoder.layers.21', 'encoder.layers.22', 'encoder.layers.23'],
        6: ['encoder'],
        10: []  # 解冻所有层
    },
    
    # 数据增强 - 加强版
    'use_spec_augment': True,      
    'freq_mask_param': 20,              
    'time_mask_param': 70,              
    'num_freq_mask': 2,
    'num_time_mask': 2,
    'speed_perturb': True,         
    'speed_rates': [0.9, 0.95, 1.0, 1.05, 1.1],
    'noise_augment': True,         
    'noise_prob': 0.3,                  
    'noise_level': 0.005,
    
    # 早停和模型选择
    'early_stopping_patience': 7,       
    'early_stopping_min_delta': 0.0005,
    'reduce_lr_patience': 3,            
    'reduce_lr_factor': 0.5,            
    
    # 其他配置
    'mixed_precision': True,            
    'gradient_checkpointing': True,
    'eval_steps': 200,                  
    'save_steps': 400,
    'logging_steps': 20,
    'num_workers': 4,
    'max_loss_value': 50.0,
    'max_audio_length': 20.0,
    'sample_rate': 16000,
    'audio_normalize': True,
    'min_audio_length': 1.0,
    'max_target_length': 256,
}

class SpecAugment:
    """SpecAugment数据增强"""
    def __init__(self, freq_mask_param, time_mask_param, num_freq_mask, num_time_mask):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_mask = num_freq_mask
        self.num_time_mask = num_time_mask
    
    def __call__(self, spectrogram):
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
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def speed_perturb(self, waveform, speed_rate):
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
    """LibriSpeech数据集类"""
    def __init__(self, data_dirs, processor, hyperparameters, is_train=True):
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.processor = processor
        self.hyperparameters = hyperparameters
        self.is_train = is_train
        
        self.max_audio_length = hyperparameters['max_audio_length']
        self.min_audio_length = hyperparameters['min_audio_length']
        self.sample_rate = hyperparameters['sample_rate']
        self.audio_normalize = hyperparameters['audio_normalize']
        self.max_target_length = hyperparameters['max_target_length']
        self.max_length_samples = int(self.max_audio_length * self.sample_rate)
        self.min_length_samples = int(self.min_audio_length * self.sample_rate)
        
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
        for data_dir in self.data_dirs:
            print(f"正在加载数据集: {data_dir}")
            self._load_directory(data_dir)
        print(f"数据集加载完成: 共 {len(self.audio_files)} 个样本")
    
    def _load_directory(self, data_dir):
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
        if self.audio_normalize:
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            rms = np.sqrt(np.mean(audio_array ** 2))
            if rms > 1e-8:
                target_rms = 0.1
                audio_array = audio_array * (target_rms / rms)
            
            audio_array = np.clip(audio_array, -1.0, 1.0)
            
        return audio_array.astype(np.float32)
    
    def _apply_augmentation(self, waveform):
        if not self.is_train or self.audio_aug is None:
            return waveform
        
        if self.hyperparameters['speed_perturb'] and random.random() < 0.5:
            speed_rate = random.choice(self.hyperparameters['speed_rates'])
            if speed_rate != 1.0:
                waveform = self.audio_aug.speed_perturb(waveform, speed_rate)
        
        if self.hyperparameters['noise_augment'] and random.random() < self.hyperparameters['noise_prob']:
            waveform = self.audio_aug.add_noise(waveform, self.hyperparameters['noise_level'])
        
        return waveform
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        transcription = self.transcriptions[idx]
        
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveform = self._apply_augmentation(waveform)
            audio_array = waveform.squeeze().numpy()
            
            if len(audio_array) > self.max_length_samples:
                if self.is_train:
                    start = np.random.randint(0, len(audio_array) - self.max_length_samples + 1)
                else:
                    start = (len(audio_array) - self.max_length_samples) // 2
                audio_array = audio_array[start:start + self.max_length_samples]
            elif len(audio_array) < self.min_length_samples:
                audio_array = np.pad(audio_array, (0, self.min_length_samples - len(audio_array)))
            
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
    """CTC数据整理器"""
    processor: any
    padding: bool = True
    spec_augment: Optional[SpecAugment] = None
    label_smoothing: float = 0.0
    
    def __call__(self, features: List[Dict[str, Union[np.ndarray, str, int, bool]]]) -> Dict[str, torch.Tensor]:
        valid_features = []
        for feature in features:
            if feature["transcription"] != "EMPTY" and len(feature["transcription"].strip()) > 0:
                valid_features.append(feature)
        
        if not valid_features:
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
        
        inputs = self.processor(
            audio_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        if torch.any(torch.isnan(inputs.input_values)) or torch.any(torch.isinf(inputs.input_values)):
            inputs.input_values = torch.nan_to_num(inputs.input_values, nan=0.0, posinf=1.0, neginf=-1.0)
        
        with self.processor.as_target_processor():
            labels = self.processor(
                text_list,
                return_tensors="pt",
                padding=True
            )
        
        labels_tensor = labels.input_ids.clone()
        batch_size = inputs.input_values.shape[0]
        
        if hasattr(inputs, 'attention_mask'):
            input_lengths = torch.sum(inputs.attention_mask, dim=1)
            estimated_output_lengths = input_lengths // 320 + 10
        else:
            input_lengths = torch.full((batch_size,), inputs.input_values.shape[1], dtype=torch.long)
            estimated_output_lengths = input_lengths // 320 + 10
        
        for i in range(batch_size):
            label = labels_tensor[i]
            valid_length = (label != self.processor.tokenizer.pad_token_id).sum().item()
            
            if valid_length > 0 and valid_length > estimated_output_lengths[i] * 0.8:
                safe_length = int(estimated_output_lengths[i].item() * 0.5)
                labels_tensor[i, safe_length:] = self.processor.tokenizer.pad_token_id
        
        labels_tensor[labels_tensor == self.processor.tokenizer.pad_token_id] = -100
        
        if torch.all(labels_tensor == -100):
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
    """计算CTC损失"""
    batch_size, max_time, vocab_size = logits.shape
    device = logits.device
    
    blank_id = processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else 0
    
    if attention_mask is not None:
        input_lengths = (torch.sum(attention_mask, dim=1) // 320).long()
        input_lengths = torch.clamp(input_lengths, min=1, max=max_time)
    else:
        input_lengths = torch.full((batch_size,), max_time, dtype=torch.long, device=device)
    
    target_lengths = []
    labels_list = []
    
    for i in range(batch_size):
        label = labels[i]
        valid_mask = label != -100
        valid_labels = label[valid_mask]
        
        if len(valid_labels) == 0:
            valid_labels = torch.tensor([blank_id], device=device)
            target_lengths.append(1)
        else:
            max_label_length = input_lengths[i].item() - 1
            if len(valid_labels) > max_label_length:
                valid_labels = valid_labels[:max_label_length]
            
            target_lengths.append(len(valid_labels))
        
        labels_list.append(valid_labels)
    
    max_target_length = max(target_lengths) if target_lengths else 1
    padded_labels = torch.full((batch_size, max_target_length), blank_id, dtype=torch.long, device=device)
    
    for i, label in enumerate(labels_list):
        if len(label) > 0:
            padded_labels[i, :len(label)] = label
    
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    log_probs = F.log_softmax(logits, dim=-1)
    
    if torch.any(torch.isnan(log_probs)) or torch.any(torch.isinf(log_probs)):
        log_probs = torch.nan_to_num(log_probs, nan=-10.0, posinf=10.0, neginf=-10.0)
    
    try:
        loss = F.ctc_loss(
            log_probs.transpose(0, 1),
            padded_labels,
            input_lengths,
            target_lengths,
            blank=blank_id,
            reduction='mean',
            zero_infinity=True
        )
        
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(10.0, device=device, requires_grad=True)
        
        if loss.item() > finetune_hyperparameters['max_loss_value']:
            loss = torch.clamp(loss, max=finetune_hyperparameters['max_loss_value'])
        
        return loss
        
    except Exception as e:
        print(f"CTC损失计算出错: {e}")
        return torch.tensor(10.0, device=device, requires_grad=True)

def setup_layer_freezing(model, epoch, hyperparameters):
    """设置层冻结策略"""
    for param in model.parameters():
        param.requires_grad = False
    
    layers_to_unfreeze = []
    for schedule_epoch, layers in sorted(hyperparameters['unfreeze_schedule'].items()):
        if epoch >= schedule_epoch:
            layers_to_unfreeze = layers
    
    if layers_to_unfreeze == []:
        for param in model.parameters():
            param.requires_grad = True
        print(f"Epoch {epoch}: 所有层已解冻")
    else:
        for name, param in model.named_parameters():
            if "lm_head" in name and "lm_head" in layers_to_unfreeze:
                param.requires_grad = True
            elif any(layer in name for layer in layers_to_unfreeze if layer != "lm_head"):
                param.requires_grad = True
        print(f"Epoch {epoch}: 解冻层 {layers_to_unfreeze}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

def apply_regularization(model, hyperparameters):
    """应用正则化设置"""
    model.config.hidden_dropout = hyperparameters['hidden_dropout']
    model.config.attention_dropout = hyperparameters['attention_dropout']
    model.config.feat_proj_dropout = hyperparameters['feat_proj_dropout']
    model.config.final_dropout = hyperparameters['dropout']
    
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = hyperparameters['dropout']
    
    return model

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
                
                if torch.any(torch.isnan(input_values)) or torch.any(torch.isinf(input_values)):
                    continue
                
                outputs = model(input_values=input_values)
                loss = compute_ctc_loss_with_length_check(
                    outputs.logits, labels, processor, attention_mask
                )
                
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > finetune_hyperparameters['max_loss_value']:
                    continue
                
                total_loss += loss.item()
                num_batches += 1
                
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                
                for pred_ids in predicted_ids:
                    pred_str = processor.decode(pred_ids, skip_special_tokens=True)
                    predictions.append(pred_str)
                
                references.extend(batch["transcriptions"])
                
            except Exception as e:
                print(f"评估批次 {batch_idx} 时出错: {e}")
                continue
    
    if num_batches == 0:
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

def train_with_finetune(model, train_dataloader, val_dataloader, processor, hyperparameters, device, save_dir, best_checkpoint_dir):
    """微调训练主循环"""
    
    # 应用正则化
    model = apply_regularization(model, hyperparameters)
    
    # 启用梯度检查点
    if hyperparameters['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        print("✓ 梯度检查点已启用")
    
    # 创建优化器
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
        betas=(0.9, 0.98),
        eps=1e-6
    )
    
    # 计算总步数
    num_training_steps = len(train_dataloader) * hyperparameters['num_epochs'] // hyperparameters['gradient_accumulation_steps']
    
    # 创建调度器
    if hyperparameters['lr_scheduler'] == 'cosine_warm_restarts':
        steps_per_epoch = len(train_dataloader) // hyperparameters['gradient_accumulation_steps']
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hyperparameters['T_0'] * steps_per_epoch,
            T_mult=hyperparameters['T_mult'],
            eta_min=hyperparameters['eta_min']
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * hyperparameters['warmup_ratio']),
            num_training_steps=num_training_steps
        )
    
    # ReduceLROnPlateau
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=hyperparameters['reduce_lr_factor'],
        patience=hyperparameters['reduce_lr_patience'],
        min_lr=hyperparameters['min_learning_rate']
    )
    
    # 混合精度
    scaler = GradScaler() if hyperparameters['mixed_precision'] else None
    
    # 初始化变量
    global_step = 0
    best_val_wer = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_wer': [],
        'val_cer': [],
        'learning_rates': [],
        'epoch_numbers': []
    }
    
    # 加载baseline指标
    baseline_wer = 1.0
    if best_checkpoint_dir:
        baseline_metrics_path = os.path.join(os.path.dirname(best_checkpoint_dir), 'best_metrics.json')
        if os.path.exists(baseline_metrics_path):
            with open(baseline_metrics_path, 'r') as f:
                baseline_metrics = json.load(f)
                baseline_wer = baseline_metrics.get('best_val_wer', 1.0)
                print(f"Baseline WER: {baseline_wer:.4f}")
    
    print(f"\n开始微调训练...")
    print(f"总训练步数: {num_training_steps}, Warmup步数: {int(num_training_steps * hyperparameters['warmup_ratio'])}")
    
    # 训练循环
    for epoch in range(hyperparameters['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{hyperparameters['num_epochs']}")
        print(f"{'='*50}")
        
        # 设置层冻结
        setup_layer_freezing(model, epoch, hyperparameters)
        
        # 训练阶段
        model.train()
        epoch_loss = 0
        num_batches = 0
        num_skipped_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"训练 Epoch {epoch + 1}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            try:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                if torch.any(torch.isnan(input_values)) or torch.any(torch.isinf(input_values)):
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
                
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > hyperparameters['max_loss_value']:
                    num_skipped_batches += 1
                    optimizer.zero_grad()
                    continue
                
                # 反向传播
                if hyperparameters['mixed_precision']:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item() * hyperparameters['gradient_accumulation_steps']
                num_batches += 1
                
                # 梯度累积和更新
                if (step + 1) % hyperparameters['gradient_accumulation_steps'] == 0 or (step + 1) == len(train_dataloader):
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
                    
                    # 更新进度条
                    if num_batches > 0:
                        avg_loss = epoch_loss / num_batches
                        current_lr = optimizer.param_groups[0]['lr']
                        progress_bar.set_postfix({
                            'loss': f"{avg_loss:.4f}",
                            'lr': f"{current_lr:.2e}",
                            'grad': f"{grad_norm:.2f}",
                            'skip': num_skipped_batches
                        })
                    
            except Exception as e:
                print(f"训练步骤出错: {e}")
                optimizer.zero_grad()
                if hyperparameters['mixed_precision'] and scaler._scale is not None:
                    scaler.update()
                continue
        
        # 计算训练指标
        avg_train_loss = epoch_loss / max(num_batches, 1)
        
        # 验证
        print(f"\nEpoch {epoch + 1} 验证中...")
        val_loss, val_wer, val_cer, _, _ = evaluate_model(
            model, val_dataloader, processor, device, use_amp=hyperparameters['mixed_precision']
        )
        
        # 更新历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_wer'].append(val_wer)
        history['val_cer'].append(val_cer)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        history['epoch_numbers'].append(epoch + 1)
        
        # 使用plateau scheduler
        plateau_scheduler.step(val_wer)
        
        # 打印结果
        print(f"\nEpoch {epoch + 1} 结果:")
        print(f"训练损失: {avg_train_loss:.4f}")
        print(f"验证 - 损失: {val_loss:.4f}, WER: {val_wer:.4f}, CER: {val_cer:.4f}")
        
        # 检查是否有改进
        improved = False
        if val_wer < best_val_wer - hyperparameters['early_stopping_min_delta']:
            best_val_wer = val_wer
            patience_counter = 0
            improved = True
            
            # 保存最佳模型
            best_model_path = os.path.join(save_dir, 'best_finetuned_model')
            model.save_pretrained(best_model_path)
            processor.save_pretrained(best_model_path)
            
            print(f"✓ 新的最佳模型! WER: {val_wer:.4f}")
            
            if val_wer < baseline_wer:
                print(f"✓ 超过baseline! 新WER: {val_wer:.4f} < Baseline: {baseline_wer:.4f}")
        else:
            patience_counter += 1
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint-epoch-{epoch+1}')
            model.save_pretrained(checkpoint_path)
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'plateau_scheduler_state_dict': plateau_scheduler.state_dict(),
                'best_val_wer': best_val_wer,
                'history': history,
                'hyperparameters': hyperparameters
            }, os.path.join(checkpoint_path, 'training_state.pt'))
        
        # 早停
        if patience_counter >= hyperparameters['early_stopping_patience']:
            print(f"早停触发 - {patience_counter} epochs没有改进")
            break
        
        print(f"最佳WER: {best_val_wer:.4f} | Patience: {patience_counter}/{hyperparameters['early_stopping_patience']}")
    
    return history, best_val_wer

def plot_finetune_curves(history, save_dir):
    """绘制微调训练曲线"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    epochs = history['epoch_numbers']
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失', marker='o')
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练损失 vs 验证损失')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. WER曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_wer'], 'g-', label='验证WER', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('WER')
    ax2.set_title('验证WER')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(history['val_wer']) * 1.1)
    
    # 3. 学习率曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['learning_rates'], 'purple', marker='o')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('学习率')
    ax3.set_title('学习率变化')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. CER曲线
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['val_cer'], 'orange', label='验证CER', marker='o')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('CER')
    ax4.set_title('验证CER')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, max(history['val_cer']) * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'finetune_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"微调训练曲线已保存到: {os.path.join(save_dir, 'finetune_curves.png')}")

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
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # 路径配置 - 请修改为您的实际路径
    BEST_MODEL_PATH = "/root/fssd/ASR_task/.cache/wav2vec2_librispeech_ctc_fixed_20250717_071203/best_model"  # 修改为您的best model路径
    
    # 创建保存目录
    save_dir = os.path.join(SAVE_PATH, f"wav2vec2_finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存超参数
    with open(os.path.join(save_dir, 'finetune_hyperparameters.json'), 'w') as f:
        json.dump(finetune_hyperparameters, f, indent=4, ensure_ascii=False)
    
    # 加载最佳模型
    print(f"\n加载最佳模型: {BEST_MODEL_PATH}")
    
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"错误: 最佳模型路径不存在: {BEST_MODEL_PATH}")
        print("请检查路径是否正确，应该指向包含best_model的目录")
        return
    
    model = Wav2Vec2ForCTC.from_pretrained(BEST_MODEL_PATH)
    processor = Wav2Vec2Processor.from_pretrained(BEST_MODEL_PATH)
    
    print(f"✅ 模型加载成功！")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    model.to(device)
    
    # 准备数据集
    print("\n准备数据集...")
    
    train_dirs = [os.path.join(DATA_PATH, "train-clean-100")]
    val_dirs = [os.path.join(DATA_PATH, "dev-clean")]
    test_dirs = [os.path.join(DATA_PATH, "test-clean")]
    
    # 创建数据增强
    spec_augment = SpecAugment(
        finetune_hyperparameters['freq_mask_param'],
        finetune_hyperparameters['time_mask_param'],
        finetune_hyperparameters['num_freq_mask'],
        finetune_hyperparameters['num_time_mask']
    ) if finetune_hyperparameters['use_spec_augment'] else None
    
    # 创建数据集
    train_dataset = LibriSpeechDataset(
        data_dirs=train_dirs,
        processor=processor,
        hyperparameters=finetune_hyperparameters,
        is_train=True
    )
    
    val_dataset = LibriSpeechDataset(
        data_dirs=val_dirs,
        processor=processor,
        hyperparameters=finetune_hyperparameters,
        is_train=False
    )
    
    test_dataset = LibriSpeechDataset(
        data_dirs=test_dirs,
        processor=processor,
        hyperparameters=finetune_hyperparameters,
        is_train=False
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据整理器
    data_collator = CTCDataCollatorWithAugment(
        processor=processor,
        spec_augment=spec_augment,
        label_smoothing=finetune_hyperparameters['label_smoothing']
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=finetune_hyperparameters['batch_size'],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=finetune_hyperparameters['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if finetune_hyperparameters['num_workers'] > 0 else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=finetune_hyperparameters['batch_size'],
        shuffle=False,
        collate_fn=data_collator,
        num_workers=finetune_hyperparameters['num_workers'],
        pin_memory=True,
        persistent_workers=True if finetune_hyperparameters['num_workers'] > 0 else False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=finetune_hyperparameters['batch_size'],
        shuffle=False,
        collate_fn=data_collator,
        num_workers=finetune_hyperparameters['num_workers'],
        pin_memory=True,
        persistent_workers=True if finetune_hyperparameters['num_workers'] > 0 else False
    )
    
    # 开始微调
    print("\n开始微调训练...")
    print(f"学习率策略: {finetune_hyperparameters['lr_scheduler']}")
    print(f"初始学习率: {finetune_hyperparameters['learning_rate']}")
    print(f"正则化: Dropout={finetune_hyperparameters['dropout']}, Weight Decay={finetune_hyperparameters['weight_decay']}")
    
    history, best_val_wer = train_with_finetune(
        model, train_dataloader, val_dataloader, processor,
        finetune_hyperparameters, device, save_dir, BEST_MODEL_PATH
    )
    
    # 保存训练历史
    with open(os.path.join(save_dir, 'finetune_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # 绘制训练曲线
    plot_finetune_curves(history, save_dir)
    
    # 最终测试集评估
    print("\n在测试集上进行最终评估...")
    
    best_finetuned_path = os.path.join(save_dir, 'best_finetuned_model')
    if os.path.exists(best_finetuned_path):
        best_model = Wav2Vec2ForCTC.from_pretrained(best_finetuned_path)
        best_model.to(device)
        
        test_loss, test_wer, test_cer, test_preds, test_refs = evaluate_model(
            best_model, test_dataloader, processor, device, 
            use_amp=finetune_hyperparameters['mixed_precision']
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
            'best_val_wer': float(best_val_wer),
            'hyperparameters': finetune_hyperparameters
        }
        
        with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # 保存预测示例
        with open(os.path.join(save_dir, 'test_predictions.txt'), 'w', encoding='utf-8') as f:
            f.write("测试集预测示例\n")
            f.write("=" * 80 + "\n\n")
            
            for i in range(min(20, len(test_preds), len(test_refs))):
                f.write(f"样本 {i+1}:\n")
                f.write(f"参考: {test_refs[i]}\n")
                f.write(f"预测: {test_preds[i]}\n")
                sample_wer = wer(test_refs[i], test_preds[i])
                f.write(f"WER: {sample_wer:.3f}\n")
                f.write("-" * 80 + "\n")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 打印总结
    print(f"\n{'='*50}")
    print("微调训练总结")
    print(f"{'='*50}")
    print(f"保存目录: {save_dir}")
    print(f"最佳验证WER: {best_val_wer:.4f}")
    print(f"训练epochs: {len(history['epoch_numbers'])}")
    
    # 计算改进
    if history['val_wer']:
        initial_wer = history['val_wer'][0]
        improvement = (initial_wer - best_val_wer) / initial_wer * 100
        print(f"WER改进: {improvement:.1f}% (从 {initial_wer:.4f} 到 {best_val_wer:.4f})")

if __name__ == "__main__":
    main()