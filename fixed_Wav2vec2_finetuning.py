#!/usr/bin/env python3
"""
修复版 Wav2Vec2 LibriSpeech训练脚本
主要修复：
1. 修复绘图时数据长度不匹配问题
2. 优化训练参数和学习率调度
3. 修复epoch处理逻辑
4. 改进数据验证和错误处理
5. 优化内存使用和训练稳定性
"""

import os
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    get_linear_schedule_with_warmup
)
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import torch.nn.functional as F
from tqdm import tqdm
from jiwer import wer, cer
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置环境变量 - 多种方式确保镜像生效
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = './huggingface_cache'
os.environ['HF_HOME'] = './huggingface_cache'

# 如果在中国大陆，使用镜像
import transformers
transformers.utils.hub.HF_HUB_OFFLINE = False

# 尝试设置镜像URL
try:
    from huggingface_hub import login
    # 不需要登录，但可以设置镜像
    pass
except ImportError:
    pass

# 优化后的超参数配置
hyperparameters = {
    'learning_rate': 1e-5,      # 适中的学习率
    'batch_size': 8,            # 增加batch size以提高稳定性
    'num_epochs': 15,           # 增加epoch数
    'weight_decay': 1e-6,       # 适中的权重衰减
    'max_grad_norm': 1.0,       # 放宽梯度裁剪
    'warmup_steps': 1000,       # 减少warmup步数
    'gradient_accumulation_steps': 4,  # 减少梯度累积
    'eval_steps': 1000,         # 增加评估间隔
    'save_steps': 2000,         # 增加保存间隔
    'logging_steps': 100,       # 增加日志间隔
    'early_stopping_patience': 3,  # 减少早停patience
    'max_audio_length': 15.0,   # 增加最大音频长度
    'sample_rate': 16000,
    'audio_normalize': True,
    'min_audio_length': 1.0,    # 减少最小音频长度
    'max_target_length': 512,   # 增加最大标签长度
}

def freeze_wav2vec2_layers(model):
    """冻结Wav2Vec2模型的所有层，除了最后的线性层（lm_head）"""
    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 只解冻lm_head
    for param in model.lm_head.parameters():
        param.requires_grad = True
    
    # 打印冻结信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n模型参数冻结情况:")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"冻结参数数量: {frozen_params:,}")
    print(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")
    
    # 验证只有lm_head是可训练的
    print("\n可训练的层:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}: {param.shape}")
    
    return model

class LibriSpeechDataset(Dataset):
    """优化后的LibriSpeech数据集类"""
    
    def __init__(self, data_dir, processor, max_audio_length=20.0, min_audio_length=1.0, 
                 sample_rate=16000, audio_normalize=True, max_target_length=512):
        self.data_dir = data_dir
        self.processor = processor
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.sample_rate = sample_rate
        self.audio_normalize = audio_normalize
        self.max_target_length = max_target_length
        self.max_length_samples = int(max_audio_length * sample_rate)
        self.min_length_samples = int(min_audio_length * sample_rate)
        
        self.audio_files = []
        self.transcriptions = []
        self._prepare_data()
    
    def _prepare_data(self):
        """准备数据集，添加更严格的过滤"""
        print(f"正在加载数据集: {self.data_dir}")
        count = 0
        filtered_count = 0
        length_filtered = 0
        
        for root, dirs, files in os.walk(self.data_dir):
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
                                
                                if os.path.exists(audio_file):
                                    # 验证转录文本
                                    if transcription.strip() and len(transcription.strip()) > 0:
                                        clean_transcription = ' '.join(transcription.upper().split())
                                        
                                        # 检查转录长度
                                        try:
                                            encoded_length = len(self.processor.tokenizer.encode(clean_transcription))
                                            if encoded_length > self.max_target_length:
                                                length_filtered += 1
                                                continue
                                        except Exception:
                                            filtered_count += 1
                                            continue
                                        
                                        # 预检查音频长度
                                        try:
                                            waveform, sr = torchaudio.load(audio_file)
                                            duration = waveform.shape[1] / sr
                                            
                                            # 过滤太短或太长的音频
                                            if duration < self.min_audio_length or duration > self.max_audio_length:
                                                length_filtered += 1
                                                continue
                                            
                                            # 估算CTC长度关系（更保守的估计）
                                            estimated_output_length = int(duration * self.sample_rate / 320)
                                            if estimated_output_length < encoded_length * 0.5:  # 更宽松的条件
                                                length_filtered += 1
                                                continue
                                                
                                        except Exception:
                                            filtered_count += 1
                                            continue
                                        
                                        self.audio_files.append(audio_file)
                                        self.transcriptions.append(clean_transcription)
                                        count += 1
                                    else:
                                        filtered_count += 1
                                    
                                    if count % 1000 == 0:
                                        print(f"已加载 {count} 个文件，过滤 {filtered_count} 个无效样本，{length_filtered} 个长度不匹配样本...")
                except Exception as e:
                    print(f"读取转录文件出错 {trans_path}: {e}")
                    continue
        
        print(f"数据集加载完成:")
        print(f"  有效样本: {len(self.audio_files)}")
        print(f"  过滤的无效样本: {filtered_count}")
        print(f"  过滤的长度不匹配样本: {length_filtered}")
    
    def _normalize_audio(self, audio_array):
        """改进的音频归一化策略"""
        if self.audio_normalize:
            # 检查并清理异常值
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 计算RMS并归一化
            rms = np.sqrt(np.mean(audio_array ** 2))
            if rms > 1e-8:
                # 归一化到合适的RMS值
                target_rms = 0.1
                audio_array = audio_array * (target_rms / rms)
            
            # 防止过大的值
            audio_array = np.clip(audio_array, -1.0, 1.0)
            
        return audio_array.astype(np.float32)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        transcription = self.transcriptions[idx]
        
        try:
            # 加载音频
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # 重采样到16kHz
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 获取音频数组
            audio_array = waveform.squeeze().numpy()
            
            # 处理音频长度
            if len(audio_array) > self.max_length_samples:
                # 随机裁剪而不是简单截断
                start = np.random.randint(0, len(audio_array) - self.max_length_samples + 1)
                audio_array = audio_array[start:start + self.max_length_samples]
            elif len(audio_array) < self.min_length_samples:
                # 用零填充
                audio_array = np.pad(audio_array, (0, self.min_length_samples - len(audio_array)))
            
            # 归一化音频
            audio_array = self._normalize_audio(audio_array)
            
            # 计算理论输出长度
            theoretical_output_length = len(audio_array) // 320
            
            return {
                "audio": audio_array,
                "transcription": transcription,
                "audio_length": len(audio_array),
                "theoretical_output_length": theoretical_output_length
            }
            
        except Exception as e:
            print(f"加载音频文件出错 {audio_file}: {e}")
            # 返回一个有效的空样本
            return {
                "audio": np.zeros(self.min_length_samples, dtype=np.float32),
                "transcription": "EMPTY",
                "audio_length": self.min_length_samples,
                "theoretical_output_length": self.min_length_samples // 320
            }

@dataclass
class CTC_SafeDataCollator:
    """CTC安全的数据整理器"""
    processor: any
    padding: bool = True
    
    def __call__(self, features: List[Dict[str, Union[np.ndarray, str, int]]]) -> Dict[str, torch.Tensor]:
        # 过滤无效样本
        valid_features = []
        for feature in features:
            if (feature["transcription"] != "EMPTY" and 
                len(feature["transcription"].strip()) > 0 and
                not np.any(np.isnan(feature["audio"])) and
                not np.any(np.isinf(feature["audio"]))):
                
                try:
                    encoded_length = len(self.processor.tokenizer.encode(feature["transcription"]))
                    theoretical_output_length = feature["theoretical_output_length"]
                    
                    # 更宽松的长度检查
                    if theoretical_output_length >= encoded_length * 0.5:
                        valid_features.append(feature)
                except Exception:
                    continue
        
        if not valid_features:
            # 创建虚拟批次
            dummy_audio = np.zeros(16000, dtype=np.float32)
            dummy_text = "HELLO"
            
            return {
                "input_values": torch.tensor([dummy_audio], dtype=torch.float32),
                "labels": torch.tensor([[self.processor.tokenizer.encode(dummy_text)[0]]], dtype=torch.long),
                "transcriptions": [dummy_text]
            }
        
        # 处理音频
        audio_list = [f["audio"] for f in valid_features]
        text_list = [f["transcription"] for f in valid_features]
        
        # 使用processor处理音频
        inputs = self.processor(
            audio_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # 处理标签
        with self.processor.as_target_processor():
            labels = self.processor(
                text_list,
                return_tensors="pt",
                padding=True
            )
        
        # 将padding位置设为-100
        labels["input_ids"][labels["input_ids"] == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "input_values": inputs.input_values,
            "labels": labels.input_ids,
            "transcriptions": text_list
        }

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
        
        # 确保返回有效的数值
        if np.isnan(word_error_rate) or np.isinf(word_error_rate):
            word_error_rate = 1.0
        if np.isnan(char_error_rate) or np.isinf(char_error_rate):
            char_error_rate = 1.0
            
    except Exception as e:
        print(f"计算WER/CER时出错: {e}")
        return 1.0, 1.0
    
    return word_error_rate, char_error_rate

def evaluate_model(model, dataloader, processor, device):
    """评估模型"""
    model.eval()
    predictions = []
    references = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            try:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                
                # 跳过异常输入
                if torch.any(torch.isnan(input_values)) or torch.any(torch.isinf(input_values)):
                    continue
                
                outputs = model(input_values=input_values, labels=labels)
                
                # 检查输出
                if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                    continue
                
                total_loss += outputs.loss.item()
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
                print(f"评估批次时出错: {e}")
                continue
    
    avg_loss = total_loss / max(num_batches, 1)
    word_error_rate, char_error_rate = compute_metrics(predictions, references)
    
    return avg_loss, word_error_rate, char_error_rate, predictions, references

def train_model(model, train_dataloader, val_dataloader, processor, hyperparameters, device, save_dir):
    """训练模型"""
    
    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=hyperparameters['learning_rate'],
        weight_decay=hyperparameters['weight_decay'],
        eps=1e-8
    )
    
    # 计算总训练步数
    num_training_steps = len(train_dataloader) * hyperparameters['num_epochs'] // hyperparameters['gradient_accumulation_steps']
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=hyperparameters['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # 训练历史 - 分开存储step级别和epoch级别的数据
    history = {
        'step_train_loss': [],
        'step_learning_rate': [],
        'step_gradient_norm': [],
        'step_numbers': [],
        'epoch_train_loss': [],
        'epoch_val_loss': [],
        'epoch_val_wer': [],
        'epoch_val_cer': []
    }
    
    # Early stopping
    best_val_wer = float('inf')
    patience_counter = 0
    
    # 开始训练
    global_step = 0
    
    for epoch in range(hyperparameters['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{hyperparameters['num_epochs']}")
        print(f"{'='*50}")
        
        # 训练阶段
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"训练 Epoch {epoch + 1}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            try:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                
                # 跳过异常输入
                if torch.any(torch.isnan(input_values)) or torch.any(torch.isinf(input_values)):
                    continue
                
                # 前向传播
                outputs = model(input_values=input_values, labels=labels)
                loss = outputs.loss / hyperparameters['gradient_accumulation_steps']
                
                # 跳过异常损失
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # 反向传播
                loss.backward()
                
                epoch_loss += loss.item() * hyperparameters['gradient_accumulation_steps']
                num_batches += 1
                
                # 梯度累积
                if (step + 1) % hyperparameters['gradient_accumulation_steps'] == 0:
                    # 梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparameters['max_grad_norm'])
                    
                    # 检查梯度
                    if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                        optimizer.step()
                        scheduler.step()
                        global_step += 1
                        
                        # 记录step级别的数据
                        if global_step % hyperparameters['logging_steps'] == 0:
                            current_lr = scheduler.get_last_lr()[0]
                            history['step_train_loss'].append(loss.item() * hyperparameters['gradient_accumulation_steps'])
                            history['step_learning_rate'].append(current_lr)
                            history['step_gradient_norm'].append(grad_norm.item())
                            history['step_numbers'].append(global_step)
                    
                    optimizer.zero_grad()
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * hyperparameters['gradient_accumulation_steps']:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                    
            except Exception as e:
                print(f"训练步骤出错: {e}")
                continue
        
        # 计算epoch平均损失
        avg_train_loss = epoch_loss / max(num_batches, 1)
        history['epoch_train_loss'].append(avg_train_loss)
        
        # 验证
        print(f"Epoch {epoch + 1} 验证中...")
        val_loss, val_wer, val_cer, val_preds, val_refs = evaluate_model(
            model, val_dataloader, processor, device
        )
        
        # 记录epoch级别的数据
        history['epoch_val_loss'].append(val_loss)
        history['epoch_val_wer'].append(val_wer)
        history['epoch_val_cer'].append(val_cer)
        
        print(f"\nEpoch {epoch + 1} 结果:")
        print(f"训练损失: {avg_train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证WER: {val_wer:.4f}")
        print(f"验证CER: {val_cer:.4f}")
        
        # 保存最佳模型
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            patience_counter = 0
            
            best_model_path = os.path.join(save_dir, 'best_model')
            model.save_pretrained(best_model_path)
            processor.save_pretrained(best_model_path)
            
            print(f"✓ 保存最佳模型 (WER: {best_val_wer:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= hyperparameters['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    return history

def plot_training_curves(history, save_dir):
    """修复后的绘图函数"""
    plt.style.use('default')
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Step级别的训练损失
    if history['step_train_loss']:
        axes[0, 0].plot(history['step_numbers'], history['step_train_loss'], 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].set_xlabel('训练步数')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('训练损失 (Step级别)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
    else:
        axes[0, 0].text(0.5, 0.5, '无训练损失数据', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('训练损失 (Step级别)')
    
    # 2. 梯度范数
    if history['step_gradient_norm']:
        axes[0, 1].plot(history['step_numbers'], history['step_gradient_norm'], 'red', alpha=0.7)
        axes[0, 1].set_xlabel('训练步数')
        axes[0, 1].set_ylabel('梯度范数')
        axes[0, 1].set_title('梯度范数')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=hyperparameters['max_grad_norm'], color='red', linestyle='--', alpha=0.5, label='裁剪阈值')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, '无梯度范数数据', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('梯度范数')
    
    # 3. 学习率
    if history['step_learning_rate']:
        axes[0, 2].plot(history['step_numbers'], history['step_learning_rate'], 'purple', alpha=0.8)
        axes[0, 2].set_xlabel('训练步数')
        axes[0, 2].set_ylabel('学习率')
        axes[0, 2].set_title('学习率调度')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, '无学习率数据', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('学习率调度')
    
    # 4. Epoch级别的损失对比
    if history['epoch_train_loss'] and history['epoch_val_loss']:
        epochs = range(1, len(history['epoch_train_loss']) + 1)
        axes[1, 0].plot(epochs, history['epoch_train_loss'], 'b-', label='训练', marker='o')
        
        # 确保验证损失长度匹配
        val_epochs = range(1, len(history['epoch_val_loss']) + 1)
        axes[1, 0].plot(val_epochs, history['epoch_val_loss'], 'r-', label='验证', marker='s')
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].set_title('训练 vs 验证损失')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    else:
        axes[1, 0].text(0.5, 0.5, '无Epoch损失数据', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('训练 vs 验证损失')
    
    # 5. 验证WER
    if history['epoch_val_wer']:
        epochs = range(1, len(history['epoch_val_wer']) + 1)
        axes[1, 1].plot(epochs, history['epoch_val_wer'], 'g-', marker='o', markersize=6)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('WER')
        axes[1, 1].set_title('验证 WER')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 设置y轴范围
        min_wer = min(history['epoch_val_wer'])
        max_wer = max(history['epoch_val_wer'])
        axes[1, 1].set_ylim(max(0, min_wer - 0.05), min(1, max_wer + 0.05))
    else:
        axes[1, 1].text(0.5, 0.5, '无WER数据', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('验证 WER')
    
    # 6. 验证CER
    if history['epoch_val_cer']:
        epochs = range(1, len(history['epoch_val_cer']) + 1)
        axes[1, 2].plot(epochs, history['epoch_val_cer'], 'm-', marker='o', markersize=6)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('CER')
        axes[1, 2].set_title('验证 CER')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 设置y轴范围
        min_cer = min(history['epoch_val_cer'])
        max_cer = max(history['epoch_val_cer'])
        axes[1, 2].set_ylim(max(0, min_cer - 0.05), min(1, max_cer + 0.05))
    else:
        axes[1, 2].text(0.5, 0.5, '无CER数据', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('验证 CER')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: {os.path.join(save_dir, 'training_curves.png')}")

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建保存目录
    save_dir = f"wav2vec2_librispeech_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存超参数
    with open(os.path.join(save_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    
    # 检查本地模型
    local_model_path = "./models/wav2vec2-large-960h"
    model_name = "facebook/wav2vec2-large-960h"
    
    # 优先使用本地模型
    if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "config.json")):
        print(f"发现本地模型: {local_model_path}")
        model_name = local_model_path
    else:
        print(f"本地模型不存在，将从在线下载: {model_name}")
        print("如果下载失败，请先运行下载脚本")
    
    # 加载模型和处理器 - 添加镜像支持
    print(f"加载模型: {model_name}")
    
    # 多种方式尝试加载模型
    def load_model_with_mirror():
        """使用多种方式加载模型"""
        
        # 方法1：如果是本地路径，直接加载
        if os.path.exists(model_name):
            try:
                print("从本地路径加载...")
                processor = Wav2Vec2Processor.from_pretrained(model_name)
                model = Wav2Vec2ForCTC.from_pretrained(model_name)
                return processor, model
            except Exception as e:
                print(f"本地路径加载失败: {e}")
        
        # 方法2：尝试从镜像加载
        try:
            print("尝试从HF镜像加载...")
            # 设置镜像环境变量
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            os.environ['HUGGINGFACE_HUB_ENDPOINT'] = 'https://hf-mirror.com'
            
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2ForCTC.from_pretrained(model_name)
            return processor, model
        except Exception as e:
            print(f"镜像加载失败: {e}")
        
        # 方法3：尝试从本地缓存加载
        try:
            print("尝试从本地缓存加载...")
            processor = Wav2Vec2Processor.from_pretrained(
                model_name, 
                local_files_only=True
            )
            model = Wav2Vec2ForCTC.from_pretrained(
                model_name, 
                local_files_only=True
            )
            return processor, model
        except Exception as e:
            print(f"本地缓存加载失败: {e}")
        
        # 方法4：尝试直接连接
        try:
            print("尝试直接连接HuggingFace...")
            # 临时清除镜像设置
            if 'HF_ENDPOINT' in os.environ:
                del os.environ['HF_ENDPOINT']
            if 'HUGGINGFACE_HUB_ENDPOINT' in os.environ:
                del os.environ['HUGGINGFACE_HUB_ENDPOINT']
                
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2ForCTC.from_pretrained(model_name)
            return processor, model
        except Exception as e:
            print(f"直接连接失败: {e}")
        
        return None, None
    
    # 执行加载
    processor, model = load_model_with_mirror()
    
    if processor is None or model is None:
        print("\n❌ 所有加载方法都失败了！")
        print("\n解决方案:")
        print("1. 运行下载脚本:")
        print("   python download_model.py")
        print("\n2. 或者手动下载:")
        print("   mkdir -p ./models/wav2vec2-large-960h")
        print("   cd ./models/wav2vec2-large-960h")
        print("   wget https://hf-mirror.com/facebook/wav2vec2-large-960h/resolve/main/config.json")
        print("   wget https://hf-mirror.com/facebook/wav2vec2-large-960h/resolve/main/preprocessor_config.json")
        print("   wget https://hf-mirror.com/facebook/wav2vec2-large-960h/resolve/main/pytorch_model.bin")
        print("   wget https://hf-mirror.com/facebook/wav2vec2-large-960h/resolve/main/special_tokens_map.json")
        print("   wget https://hf-mirror.com/facebook/wav2vec2-large-960h/resolve/main/tokenizer_config.json")
        print("   wget https://hf-mirror.com/facebook/wav2vec2-large-960h/resolve/main/vocab.json")
        print("\n3. 使用VPN或代理连接")
        print("\n4. 或者使用其他预训练模型")
        return
    
    print(f"✅ 模型加载成功！")
    print(f"词汇表大小: {len(processor.tokenizer.vocab)}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    model.to(device)
    
    # 冻结特征提取器以提高训练稳定性
    model.freeze_feature_encoder()
    
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 准备数据集
    print("\n准备数据集...")
    
    # 检查数据路径
    data_paths = {
        'train': "/root/fssd/huggingface/datasets/LibriSpeech/train-clean-100",
        'val': "/root/fssd/huggingface/datasets/LibriSpeech/dev-clean",
        'test': "/root/fssd/huggingface/datasets/LibriSpeech/test-clean"
    }
    
    # 验证数据路径
    for name, path in data_paths.items():
        if not os.path.exists(path):
            print(f"警告: {name} 数据路径不存在: {path}")
            # 尝试替代路径
            alt_path = path.replace("/root/fssd/huggingface/datasets/LibriSpeech/", "./LibriSpeech/")
            if os.path.exists(alt_path):
                data_paths[name] = alt_path
                print(f"使用替代路径: {alt_path}")
            else:
                print(f"无法找到 {name} 数据，请检查路径")
                return
    
    # 创建数据集
    train_dataset = LibriSpeechDataset(
        data_dir=data_paths['train'],
        processor=processor,
        max_audio_length=hyperparameters['max_audio_length'],
        min_audio_length=hyperparameters['min_audio_length'],
        audio_normalize=hyperparameters['audio_normalize'],
        max_target_length=hyperparameters['max_target_length']
    )
    
    val_dataset = LibriSpeechDataset(
        data_dir=data_paths['val'],
        processor=processor,
        max_audio_length=hyperparameters['max_audio_length'],
        min_audio_length=hyperparameters['min_audio_length'],
        audio_normalize=hyperparameters['audio_normalize'],
        max_target_length=hyperparameters['max_target_length']
    )
    
    test_dataset = LibriSpeechDataset(
        data_dir=data_paths['test'],
        processor=processor,
        max_audio_length=hyperparameters['max_audio_length'],
        min_audio_length=hyperparameters['min_audio_length'],
        audio_normalize=hyperparameters['audio_normalize'],
        max_target_length=hyperparameters['max_target_length']
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("错误: 数据集为空，请检查数据路径和格式")
        return
    
    # 创建数据整理器
    data_collator = CTC_SafeDataCollator(processor=processor)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 测试一个batch
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
    
    history = train_model(
        model, train_dataloader, val_dataloader,
        processor, hyperparameters, device, save_dir
    )
    
    # 保存训练历史
    print("\n保存训练历史...")
    try:
        # 确保所有数据都是可序列化的
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, list):
                # 转换为浮点数，处理可能的numpy类型
                history_serializable[key] = [float(x) if isinstance(x, (int, float, np.number)) else x for x in value]
            else:
                history_serializable[key] = value
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history_serializable, f, indent=4)
        
        print("✓ 训练历史保存成功")
    except Exception as e:
        print(f"保存训练历史失败: {e}")
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    try:
        plot_training_curves(history, save_dir)
        print("✓ 训练曲线绘制成功")
    except Exception as e:
        print(f"绘制训练曲线失败: {e}")
    
    # 在测试集上评估
    print("\n在测试集上进行最终评估...")
    
    best_model_path = os.path.join(save_dir, 'best_model')
    if os.path.exists(best_model_path):
        try:
            # 加载最佳模型
            best_model = Wav2Vec2ForCTC.from_pretrained(best_model_path)
            best_model.to(device)
            
            test_loss, test_wer, test_cer, test_preds, test_refs = evaluate_model(
                best_model, test_dataloader, processor, device
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
                'best_val_wer': float(min(history['epoch_val_wer'])) if history['epoch_val_wer'] else float('inf'),
                'best_val_cer': float(min(history['epoch_val_cer'])) if history['epoch_val_cer'] else float('inf'),
                'total_epochs': len(history['epoch_train_loss']),
                'total_steps': len(history['step_train_loss']),
                'hyperparameters': hyperparameters
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
                    f.write("-" * 80 + "\n")
            
            print("✓ 测试结果保存成功")
            
        except Exception as e:
            print(f"测试集评估失败: {e}")
    else:
        print("未找到最佳模型，跳过测试集评估")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n训练完成！")
    print(f"所有结果保存在: {save_dir}")
    print(f"最佳验证WER: {min(history['epoch_val_wer']):.4f}" if history['epoch_val_wer'] else "无验证数据")
    
    # 提供一些后续建议
    print("\n训练建议:")
    if history['epoch_val_wer'] and len(history['epoch_val_wer']) > 1:
        final_wer = history['epoch_val_wer'][-1]
        best_wer = min(history['epoch_val_wer'])
        
        if final_wer > best_wer * 1.1:
            print("- 验证WER在上升，可能出现过拟合，建议减少epoch数或增加正则化")
        elif best_wer > 0.5:
            print("- WER较高，建议增加训练epoch数或调整超参数")
        elif best_wer < 0.1:
            print("- WER较低，模型训练效果良好！")
    
    if history['step_train_loss']:
        recent_loss = history['step_train_loss'][-10:]
        if len(recent_loss) > 5:
            loss_trend = np.mean(recent_loss[-5:]) - np.mean(recent_loss[:5])
            if loss_trend > 0:
                print("- 训练损失呈上升趋势，建议检查学习率或梯度裁剪设置")

if __name__ == "__main__":
    main()
