
#!/usr/bin/env python3
"""
完整修复版 Wav2Vec2 LibriSpeech训练脚本
主要修复：
1. 修复CTC长度不匹配问题（关键修复）
2. 删除重复的前向传播调用
3. 优化音频归一化策略
4. 进一步降低学习率
5. 添加详细的调试信息
6. 改进数据处理的鲁棒性
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




# 设置环境变量（使用镜像）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 修复后的超参数配置
hyperparameters = {
    'learning_rate': 5e-6,  # 进一步降低学习率，防止梯度爆炸
    'batch_size': 4,        # 进一步减小batch size以降低复杂性
    'num_epochs': 10,
    'weight_decay': 1e-7,   # 进一步降低权重衰减
    'max_grad_norm': 0.1,   # 更严格的梯度裁剪
    'warmup_steps': 2000,   # 更多warmup步数
    'gradient_accumulation_steps': 16,  # 增加梯度累积来补偿小batch size
    'eval_steps': 500,
    'save_steps': 1000,
    'logging_steps': 50,
    'early_stopping_patience': 5,
    'max_audio_length': 10.0,  # 进一步减少最大音频长度
    'sample_rate': 16000,
    'audio_normalize': True,
    'min_audio_length': 2.0,   # 增加最小音频长度
    'max_target_length': 256,  # 添加最大标签长度限制
}

class LibriSpeechDataset(Dataset):
    """修复后的LibriSpeech数据集类，重点解决CTC长度问题"""
    
    def __init__(self, data_dir, processor, max_audio_length=20.0, min_audio_length=2.0, 
                 sample_rate=16000, audio_normalize=True, max_target_length=256):
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
                                    
                                    # 检查转录长度（关键过滤）
                                    encoded_length = len(self.processor.tokenizer.encode(clean_transcription))
                                    if encoded_length > self.max_target_length:
                                        length_filtered += 1
                                        continue
                                    
                                    # 预检查音频长度
                                    try:
                                        waveform, sr = torchaudio.load(audio_file)
                                        duration = waveform.shape[1] / sr
                                        
                                        # 过滤太短或太长的音频
                                        if duration < self.min_audio_length or duration > self.max_audio_length:
                                            length_filtered += 1
                                            continue
                                        
                                        # 估算CTC长度关系
                                        # Wav2Vec2的输出序列长度约为输入的1/320（经验值）
                                        estimated_output_length = int(duration * self.sample_rate / 320)
                                        if estimated_output_length < encoded_length:
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
        
        print(f"数据集加载完成:")
        print(f"  有效样本: {len(self.audio_files)}")
        print(f"  过滤的无效样本: {filtered_count}")
        print(f"  过滤的长度不匹配样本: {length_filtered}")
    
    def _normalize_audio(self, audio_array):
        """改进的音频归一化策略"""
        if self.audio_normalize:
            # 检查输入数据
            if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
                print(f"警告：归一化前音频包含NaN或Inf")
                audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 计算最大值
            max_val = np.max(np.abs(audio_array))
            if max_val > 1e-8:  # 避免除零
                # 归一化到[-0.8, 0.8]，保留一些余量
                audio_array = audio_array / max_val * 0.8
            else:
                # 如果音频太小，用零填充
                audio_array = np.zeros_like(audio_array)
        
        # 最终检查
        if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
            print(f"警告：归一化后仍包含NaN或Inf，用零替换")
            audio_array = np.zeros_like(audio_array)
            
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
            if len(audio_array) < self.min_length_samples:
                # 太短的音频，用零填充
                audio_array = np.pad(audio_array, (0, self.min_length_samples - len(audio_array)))
            elif len(audio_array) > self.max_length_samples:
                # 太长的音频，截断
                audio_array = audio_array[:self.max_length_samples]
            
            # 归一化音频
            audio_array = self._normalize_audio(audio_array)
            
            # 验证音频数据
            if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
                print(f"警告：音频包含NaN或Inf值: {audio_file}")
                audio_array = np.zeros(self.min_length_samples, dtype=np.float32)
                transcription = "EMPTY"
            
            # 计算音频的理论输出长度（用于CTC检查）
            theoretical_output_length = len(audio_array) // 320  # Wav2Vec2的下采样率
            
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
    """
    CTC安全的数据整理器，重点解决长度匹配问题
    """
    processor: any
    padding: bool = True
    
    def __call__(self, features: List[Dict[str, Union[np.ndarray, str, int]]]) -> Dict[str, torch.Tensor]:
        # 过滤无效样本，增加更严格的检查
        valid_features = []
        for i, feature in enumerate(features):
            # 基本检查
            if feature["transcription"] == "EMPTY":
                print(f"跳过样本 {i}: 转录为EMPTY")
                continue
                
            if len(feature["transcription"].strip()) == 0:
                print(f"跳过样本 {i}: 转录为空")
                continue
                
            # 检查音频数据
            if np.any(np.isnan(feature["audio"])):
                print(f"跳过样本 {i}: 音频包含NaN")
                continue
                
            if np.any(np.isinf(feature["audio"])):
                print(f"跳过样本 {i}: 音频包含Inf")
                continue
                
            # 检查音频值范围
            audio_min = np.min(feature["audio"])
            audio_max = np.max(feature["audio"])
            if abs(audio_min) > 10 or abs(audio_max) > 10:
                print(f"跳过样本 {i}: 音频值超出合理范围 [{audio_min:.6f}, {audio_max:.6f}]")
                continue
                
            # CTC长度检查
            try:
                encoded_length = len(self.processor.tokenizer.encode(feature["transcription"]))
                theoretical_output_length = feature["theoretical_output_length"]
                
                # 确保音频输出长度 >= 标签长度
                if theoretical_output_length >= encoded_length:
                    valid_features.append(feature)
                    print(f"接受样本 {i}: output_len={theoretical_output_length}, target_len={encoded_length}")
                else:
                    print(f"跳过样本 {i}: 长度不匹配 output_len={theoretical_output_length}, target_len={encoded_length}")
                    
            except Exception as e:
                print(f"跳过样本 {i}: 编码检查出错: {e}")
                continue
        
        if not valid_features:
            # 如果没有有效样本，创建一个安全的虚拟批次
            print("警告：批次中没有有效样本，创建虚拟批次")
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1秒音频
            dummy_text = "A"  # 单个字符
            
            return {
                "input_values": torch.tensor([dummy_audio], dtype=torch.float32),
                "labels": torch.tensor([[self.processor.tokenizer.encode(dummy_text)[0]]], dtype=torch.long),
                "transcriptions": [dummy_text],
                "input_lengths": torch.tensor([len(dummy_audio)], dtype=torch.long),
                "target_lengths": torch.tensor([1], dtype=torch.long)
            }
        
        # 分离数据
        audio_list = [f["audio"] for f in valid_features]
        text_list = [f["transcription"] for f in valid_features]
        
        # 找到最大长度
        max_audio_length = max(len(audio) for audio in audio_list)
        
        # 手动padding音频
        padded_audio = []
        input_lengths = []
        
        for audio in audio_list:
            input_lengths.append(len(audio))
            if len(audio) < max_audio_length:
                padding = np.zeros(max_audio_length - len(audio), dtype=np.float32)
                padded = np.concatenate([audio, padding])
            else:
                padded = audio[:max_audio_length]
            padded_audio.append(padded)
        
        # 在创建张量前先检查numpy数组
        padded_audio_array = np.array(padded_audio)
        if np.any(np.isnan(padded_audio_array)) or np.any(np.isinf(padded_audio_array)):
            print("警告：padded_audio包含NaN或Inf，用零替换")
            padded_audio_array = np.nan_to_num(padded_audio_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 转换为张量并验证
        input_values = torch.tensor(padded_audio_array, dtype=torch.float32)
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        
        # 二次验证输入数据
        if torch.any(torch.isnan(input_values)) or torch.any(torch.isinf(input_values)):
            print("警告：input_values仍包含NaN或Inf，强制用零替换")
            input_values = torch.zeros_like(input_values)
        
        # 处理文本标签
        encoded_labels = []
        target_lengths = []
        
        for text in text_list:
            try:
                encoded = self.processor.tokenizer.encode(text)
                if not encoded:
                    encoded = [self.processor.tokenizer.unk_token_id]
                encoded_labels.append(encoded)
                target_lengths.append(len(encoded))
            except Exception as e:
                print(f"编码文本时出错: {text}, 错误: {e}")
                encoded_labels.append([self.processor.tokenizer.unk_token_id])
                target_lengths.append(1)
        
        # 找到最大标签长度
        max_label_length = max(len(label) for label in encoded_labels)
        
        # 手动padding标签
        padded_labels = []
        for label in encoded_labels:
            if len(label) < max_label_length:
                padding = [self.processor.tokenizer.pad_token_id] * (max_label_length - len(label))
                padded = label + padding
            else:
                padded = label[:max_label_length]
            padded_labels.append(padded)
        
        # 转换为张量
        labels = torch.tensor(padded_labels, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        
        # 将padding位置替换为-100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # 最终的CTC长度验证
        theoretical_output_lengths = input_lengths // 320  # Wav2Vec2的下采样率
        
        # 打印长度信息用于调试
        if len(valid_features) > 0:
            print(f"批次大小: {len(valid_features)}")
            print(f"输入长度范围: {torch.min(input_lengths)} - {torch.max(input_lengths)}")
            print(f"理论输出长度范围: {torch.min(theoretical_output_lengths)} - {torch.max(theoretical_output_lengths)}")
            print(f"目标长度范围: {torch.min(target_lengths)} - {torch.max(target_lengths)}")
            print(f"输入值范围: [{torch.min(input_values):.6f}, {torch.max(input_values):.6f}]")
            
            # 检查是否有NaN或Inf
            if torch.any(torch.isnan(input_values)) or torch.any(torch.isinf(input_values)):
                print("❌ 警告：最终input_values仍包含NaN或Inf！")
            else:
                print("✓ input_values检查通过")
        else:
            print("警告：没有有效特征，返回虚拟批次")
        
        return {
            "input_values": input_values,
            "labels": labels,
            "transcriptions": text_list,
            "input_lengths": input_lengths,
            "target_lengths": target_lengths
        }

def compute_metrics(predictions, references):
    """计算WER和CER"""
    valid_pairs = [(p.strip(), r.strip()) for p, r in zip(predictions, references) 
                   if r.strip() and p.strip()]
    
    if not valid_pairs:
        return 1.0, 1.0
    
    predictions, references = zip(*valid_pairs)
    
    try:
        word_error_rate = wer(list(references), list(predictions))
        char_error_rate = cer(list(references), list(predictions))
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
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                
                # 验证输入
                if torch.any(torch.isnan(input_values)) or torch.any(torch.isinf(input_values)):
                    print("警告：评估时发现NaN或Inf输入，跳过此批次")
                    continue
                
                # 前向传播
                outputs = model(input_values=input_values, labels=labels)
                loss = outputs.loss
                
                # 验证损失和输出
                if torch.isnan(loss) or torch.isinf(loss):
                    print("警告：评估时发现NaN或Inf损失，跳过此批次")
                    continue
                
                if torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any():
                    print("警告：评估时发现NaN或Inf logits，跳过此批次")
                    continue
                
                total_loss += loss.item()
                num_batches += 1
                
                # 获取预测
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_ids_np = predicted_ids.cpu().numpy()
                
                # 逐个解码
                for pred_ids in predicted_ids_np:
                    filtered_ids = [id for id in pred_ids if id != -100 and id != processor.tokenizer.pad_token_id]
                    pred_str = processor.tokenizer.decode(filtered_ids)
                    predictions.append(pred_str)
                
                references.extend(batch["transcriptions"])
                
            except Exception as e:
                print(f"评估批次时出错: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    word_error_rate, char_error_rate = compute_metrics(predictions, references)
    
    return avg_loss, word_error_rate, char_error_rate, predictions, references

def train_model(model, train_dataloader, val_dataloader, processor, hyperparameters, device, save_dir):
    """训练模型，重点修复CTC相关问题"""
    
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
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_wer': [],
        'val_cer': [],
        'learning_rate': [],
        'epoch_train_loss': [],
        'epoch_val_loss': [],
        'epoch_val_wer': [],
        'epoch_val_cer': [],
        'gradient_norm': []
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
        nan_batches = 0
        ctc_error_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            try:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                
                # 验证输入数据
                if torch.any(torch.isnan(input_values)) or torch.any(torch.isinf(input_values)):
                    print(f"警告：步骤 {step} 输入包含NaN或Inf，跳过")
                    nan_batches += 1
                    continue
                
                # 前向传播（只调用一次）
                outputs = model(input_values=input_values, labels=labels)
                
                # 检查模型输出
                if torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any():
                    print(f"警告：步骤 {step} 模型输出包含NaN或Inf，跳过")
                    nan_batches += 1
                    continue
                
                loss = outputs.loss / hyperparameters['gradient_accumulation_steps']
                
                # 验证损失
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告：步骤 {step} 损失为NaN或Inf: {loss}，跳过")
                    ctc_error_batches += 1
                    continue
                
                # 反向传播
                loss.backward()
                
                # 记录损失
                epoch_loss += loss.item() * hyperparameters['gradient_accumulation_steps']
                num_batches += 1
                
                # 梯度累积
                if (step + 1) % hyperparameters['gradient_accumulation_steps'] == 0:
                    # 检查梯度并裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparameters['max_grad_norm'])
                    
                    # 检查梯度
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 100:
                        print(f"警告：梯度范数异常: {grad_norm}，跳过优化步骤")
                        optimizer.zero_grad()
                        continue
                    
                    # 优化器步骤
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # 记录
                    if global_step % hyperparameters['logging_steps'] == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        history['learning_rate'].append(current_lr)
                        history['train_loss'].append(loss.item() * hyperparameters['gradient_accumulation_steps'])
                        history['gradient_norm'].append(grad_norm.item())
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * hyperparameters['gradient_accumulation_steps']:.6f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                        'grad_norm': f"{grad_norm:.3f}",
                        'nan_batches': nan_batches,
                        'ctc_errors': ctc_error_batches
                    })
                    
            except Exception as e:
                print(f"\n训练步骤出错: {e}")
                nan_batches += 1
                continue
        
        # 计算epoch平均损失
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        history['epoch_train_loss'].append(avg_train_loss)
        
        print(f"\nEpoch {epoch + 1} 训练统计:")
        print(f"有效批次: {num_batches}, NaN批次: {nan_batches}, CTC错误批次: {ctc_error_batches}")
        
        # 如果训练损失为NaN，提前停止
        if np.isnan(avg_train_loss) or np.isinf(avg_train_loss):
            print("训练损失为NaN或Inf，提前停止训练")
            break
        
        # Epoch结束时的验证
        print(f"\nEpoch {epoch + 1} 验证...")
        val_loss, val_wer, val_cer, val_preds, val_refs = evaluate_model(
            model, val_dataloader, processor, device
        )
        
        history['epoch_val_loss'].append(val_loss)
        history['epoch_val_wer'].append(val_wer)
        history['epoch_val_cer'].append(val_cer)
        
        print(f"\nEpoch {epoch + 1} 总结:")
        print(f"训练损失: {avg_train_loss:.6f}")
        print(f"验证损失: {val_loss:.6f}")
        print(f"验证WER: {val_wer:.4f}")
        print(f"验证CER: {val_cer:.4f}")
        
        # 保存最佳模型
        if val_wer < best_val_wer and not np.isnan(val_wer) and not np.isinf(val_wer):
            best_val_wer = val_wer
            patience_counter = 0
            
            best_model_path = os.path.join(save_dir, 'best_model')
            model.save_pretrained(best_model_path)
            processor.save_pretrained(best_model_path)
            
            print(f"✓ 保存最佳模型 (WER: {best_val_wer:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= hyperparameters['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    return history

def plot_training_curves(history, save_dir):
    """绘制训练曲线"""
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 训练损失
    if history['train_loss']:
        axes[0, 0].plot(history['train_loss'], 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].set_xlabel('Steps (x50)')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss (Step-level)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
    else:
        axes[0, 0].text(0.5, 0.5, 'No training loss data', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Training Loss (Step-level)')
    
    # 梯度范数
    if history['gradient_norm']:
        axes[0, 1].plot(history['gradient_norm'], 'red', alpha=0.7)
        axes[0, 1].set_xlabel('Steps (x50)')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].set_title('Gradient Norm')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=hyperparameters['max_grad_norm'], color='red', linestyle='--', alpha=0.5, label='Clip Threshold')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No gradient norm data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Gradient Norm')
    
    # 学习率
    if history['learning_rate']:
        axes[0, 2].plot(history['learning_rate'], 'purple', alpha=0.8)
        axes[0, 2].set_xlabel('Steps (x50)')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'No learning rate data', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Learning Rate Schedule')
    
    # Epoch级别的损失 - 修复长度不匹配问题
    if history['epoch_train_loss'] and history['epoch_val_loss']:
        epochs = range(1, min(len(history['epoch_train_loss']), len(history['epoch_val_loss'])) + 1)
        train_losses = history['epoch_train_loss'][:len(epochs)]
        val_losses = history['epoch_val_loss'][:len(epochs)]
        
        axes[1, 0].plot(epochs, train_losses, 'b-', label='Train', marker='o')
        axes[1, 0].plot(epochs, val_losses, 'r-', label='Val', marker='s')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training vs Validation Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    elif history['epoch_train_loss']:
        epochs = range(1, len(history['epoch_train_loss']) + 1)
        axes[1, 0].plot(epochs, history['epoch_train_loss'], 'b-', label='Train', marker='o')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss Only')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    else:
        axes[1, 0].text(0.5, 0.5, 'No epoch loss data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Training vs Validation Loss')
    
    # WER
    if history['epoch_val_wer']:
        epochs = range(1, len(history['epoch_val_wer']) + 1)
        axes[1, 1].plot(epochs, history['epoch_val_wer'], 'g-', marker='o', markersize=8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('WER')
        axes[1, 1].set_title('Validation WER')
        axes[1, 1].set_ylim(0, max(history['epoch_val_wer']) * 1.1)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No WER data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Validation WER')
    
    # CER
    if history['epoch_val_cer']:
        epochs = range(1, len(history['epoch_val_cer']) + 1)
        axes[1, 2].plot(epochs, history['epoch_val_cer'], 'm-', marker='o', markersize=8)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('CER')
        axes[1, 2].set_title('Validation CER')
        axes[1, 2].set_ylim(0, max(history['epoch_val_cer']) * 1.1)
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'No CER data', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Validation CER')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def debug_single_sample(processor, data_dir):
    """调试单个样本的处理过程"""
    print("\n=== 调试单个样本处理 ===")
    
    # 找第一个音频文件
    for root, dirs, files in os.walk(data_dir):
        flac_files = [f for f in files if f.endswith('.flac')]
        trans_files = [f for f in files if f.endswith('.trans.txt')]
        
        if flac_files and trans_files:
            # 读取转录文件
            trans_path = os.path.join(root, trans_files[0])
            with open(trans_path, 'r', encoding='utf-8') as f:
                line = f.readline().strip()
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    audio_id, transcription = parts
                    audio_file = os.path.join(root, f"{audio_id}.flac")
                    
                    if os.path.exists(audio_file):
                        print(f"测试文件: {audio_file}")
                        print(f"转录: {transcription}")
                        
                        try:
                            # 逐步处理
                            print("\n1. 加载音频...")
                            waveform, sample_rate = torchaudio.load(audio_file)
                            print(f"   原始形状: {waveform.shape}, 采样率: {sample_rate}")
                            print(f"   原始值范围: [{torch.min(waveform):.6f}, {torch.max(waveform):.6f}]")
                            
                            print("\n2. 重采样...")
                            if sample_rate != 16000:
                                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                                waveform = resampler(waveform)
                            print(f"   重采样后形状: {waveform.shape}")
                            print(f"   重采样后值范围: [{torch.min(waveform):.6f}, {torch.max(waveform):.6f}]")
                            
                            print("\n3. 转单声道...")
                            if waveform.shape[0] > 1:
                                waveform = torch.mean(waveform, dim=0, keepdim=True)
                            print(f"   单声道后形状: {waveform.shape}")
                            
                            print("\n4. 转numpy...")
                            audio_array = waveform.squeeze().numpy()
                            print(f"   numpy形状: {audio_array.shape}")
                            print(f"   numpy值范围: [{np.min(audio_array):.6f}, {np.max(audio_array):.6f}]")
                            print(f"   是否有NaN: {np.any(np.isnan(audio_array))}")
                            print(f"   是否有Inf: {np.any(np.isinf(audio_array))}")
                            
                            print("\n5. 归一化...")
                            max_val = np.max(np.abs(audio_array))
                            if max_val > 1e-8:
                                audio_array = audio_array / max_val * 0.8
                            print(f"   归一化后值范围: [{np.min(audio_array):.6f}, {np.max(audio_array):.6f}]")
                            print(f"   归一化后是否有NaN: {np.any(np.isnan(audio_array))}")
                            print(f"   归一化后是否有Inf: {np.any(np.isinf(audio_array))}")
                            
                            print("\n6. 转tensor...")
                            tensor = torch.tensor(audio_array, dtype=torch.float32)
                            print(f"   tensor形状: {tensor.shape}")
                            print(f"   tensor值范围: [{torch.min(tensor):.6f}, {torch.max(tensor):.6f}]")
                            print(f"   tensor是否有NaN: {torch.any(torch.isnan(tensor))}")
                            print(f"   tensor是否有Inf: {torch.any(torch.isinf(tensor))}")
                            
                            print("\n7. 测试tokenizer...")
                            encoded = processor.tokenizer.encode(transcription.upper())
                            decoded = processor.tokenizer.decode(encoded)
                            print(f"   编码长度: {len(encoded)}")
                            print(f"   编码: {encoded[:10]}...")
                            print(f"   解码: {decoded}")
                            
                            print("\n✓ 单样本处理测试完成")
                            return True
                            
                        except Exception as e:
                            print(f"❌ 处理出错: {e}")
                            return False
            break
    
def test_data_processing(processor, data_dir):
    """测试数据处理流程"""
    print("测试数据处理...")
    
    test_dataset = LibriSpeechDataset(
        data_dir=data_dir,
        processor=processor,
        max_audio_length=hyperparameters['max_audio_length'],
        min_audio_length=hyperparameters['min_audio_length'],
        audio_normalize=hyperparameters['audio_normalize'],
        max_target_length=hyperparameters['max_target_length']
    )
    
    if len(test_dataset) == 0:
        print("错误：测试数据集为空")
        return False
    
    # 测试几个样本
    for i in range(min(3, len(test_dataset))):
        sample = test_dataset[i]
        print(f"\n样本 {i+1}:")
        print(f"  音频形状: {sample['audio'].shape}")
        print(f"  转录: {sample['transcription']}")
        print(f"  音频值范围: [{np.min(sample['audio']):.4f}, {np.max(sample['audio']):.4f}]")
        print(f"  理论输出长度: {sample['theoretical_output_length']}")
        
        # 检查编码长度
        encoded_length = len(processor.tokenizer.encode(sample['transcription']))
        print(f"  编码长度: {encoded_length}")
        print(f"  长度比例: {sample['theoretical_output_length'] / encoded_length:.2f}")
        
        if np.any(np.isnan(sample['audio'])) or np.any(np.isinf(sample['audio'])):
            print("  ❌ 音频数据包含NaN或Inf值")
            return False
    
    # 测试collator
    batch = [test_dataset[i] for i in range(min(2, len(test_dataset)))]
    collator = CTC_SafeDataCollator(processor=processor)
    processed_batch = collator(batch)
    
    print(f"\n处理后的批次:")
    print(f"  input_values形状: {processed_batch['input_values'].shape}")
    print(f"  labels形状: {processed_batch['labels'].shape}")
    print(f"  转录数量: {len(processed_batch['transcriptions'])}")
    
    if torch.any(torch.isnan(processed_batch['input_values'])) or torch.any(torch.isinf(processed_batch['input_values'])):
        print("  ❌ 处理后的数据包含NaN或Inf值")
        return False
    
    print("✓ 数据处理测试通过")
    return True
    """测试数据处理流程"""
    print("测试数据处理...")
    
    test_dataset = LibriSpeechDataset(
        data_dir=data_dir,
        processor=processor,
        max_audio_length=hyperparameters['max_audio_length'],
        min_audio_length=hyperparameters['min_audio_length'],
        audio_normalize=hyperparameters['audio_normalize'],
        max_target_length=hyperparameters['max_target_length']
    )
    
    if len(test_dataset) == 0:
        print("错误：测试数据集为空")
        return False
    
    # 测试几个样本
    for i in range(min(3, len(test_dataset))):
        sample = test_dataset[i]
        print(f"\n样本 {i+1}:")
        print(f"  音频形状: {sample['audio'].shape}")
        print(f"  转录: {sample['transcription']}")
        print(f"  音频值范围: [{np.min(sample['audio']):.4f}, {np.max(sample['audio']):.4f}]")
        print(f"  理论输出长度: {sample['theoretical_output_length']}")
        
        # 检查编码长度
        encoded_length = len(processor.tokenizer.encode(sample['transcription']))
        print(f"  编码长度: {encoded_length}")
        print(f"  长度比例: {sample['theoretical_output_length'] / encoded_length:.2f}")
        
        if np.any(np.isnan(sample['audio'])) or np.any(np.isinf(sample['audio'])):
            print("  ❌ 音频数据包含NaN或Inf值")
            return False
    
    # 测试collator
    batch = [test_dataset[i] for i in range(min(2, len(test_dataset)))]
    collator = CTC_SafeDataCollator(processor=processor)
    processed_batch = collator(batch)
    
    print(f"\n处理后的批次:")
    print(f"  input_values形状: {processed_batch['input_values'].shape}")
    print(f"  labels形状: {processed_batch['labels'].shape}")
    print(f"  转录数量: {len(processed_batch['transcriptions'])}")
    
    if torch.any(torch.isnan(processed_batch['input_values'])) or torch.any(torch.isinf(processed_batch['input_values'])):
        print("  ❌ 处理后的数据包含NaN或Inf值")
        return False
    
    print("✓ 数据处理测试通过")
    return True

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建保存目录
    save_dir = f"wav2vec2_librispeech_ctc_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存超参数
    with open(os.path.join(save_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    
    # 加载模型和处理器
    model_name = "facebook/wav2vec2-large-960h"
    print(f"\n加载模型: {model_name}")
    
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        # 检查tokenizer
        print(f"词汇表大小: {len(processor.tokenizer.vocab)}")
        print(f"特殊token: {processor.tokenizer.special_tokens_map}")
        
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    model.to(device)
    
    # 检查模型参数
    nan_params = 0
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"模型参数包含NaN: {name}")
            nan_params += 1
    
    if nan_params == 0:
        print("✓ 模型参数检查通过，无NaN值")
    
    # 冻结特征提取器
    model.freeze_feature_encoder()
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 调试单个样本处理
    print("\n=== 单样本调试 ===")
    if not debug_single_sample(processor, "/root/fssd/huggingface/datasets/LibriSpeech/dev-clean"):
        print("单样本调试失败，退出")
        return
    
    # 测试数据处理
    print("\n=== 数据处理测试 ===")
    if not test_data_processing(processor, "/root/fssd/huggingface/datasets/LibriSpeech/dev-clean"):
        print("数据处理测试失败，退出")
        return
    
    # 准备数据集
    print("\n准备数据集...")
    
    train_dataset = LibriSpeechDataset(
        data_dir="/root/fssd/huggingface/datasets/LibriSpeech/train-clean-100",
        processor=processor,
        max_audio_length=hyperparameters['max_audio_length'],
        min_audio_length=hyperparameters['min_audio_length'],
        audio_normalize=hyperparameters['audio_normalize'],
        max_target_length=hyperparameters['max_target_length']
    )
    
    val_dataset = LibriSpeechDataset(
        data_dir="/root/fssd/huggingface/datasets/LibriSpeech/dev-clean",
        processor=processor,
        max_audio_length=hyperparameters['max_audio_length'],
        min_audio_length=hyperparameters['min_audio_length'],
        audio_normalize=hyperparameters['audio_normalize'],
        max_target_length=hyperparameters['max_target_length']
    )
    
    test_dataset = LibriSpeechDataset(
        data_dir="/root/fssd/huggingface/datasets/LibriSpeech/test-clean",
        processor=processor,
        max_audio_length=hyperparameters['max_audio_length'],
        min_audio_length=hyperparameters['min_audio_length'],
        audio_normalize=hyperparameters['audio_normalize'],
        max_target_length=hyperparameters['max_target_length']
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据整理器
    data_collator = CTC_SafeDataCollator(
        processor=processor,
        padding=True
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True
    )
    
    # 开始训练
    print("\n开始训练...")
    history = train_model(
        model, train_dataloader, val_dataloader,
        processor, hyperparameters, device, save_dir
    )
    
    # 保存训练历史
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, list):
                history_serializable[key] = [float(x) if not isinstance(x, str) else x for x in value]
            else:
                history_serializable[key] = value
        json.dump(history_serializable, f, indent=4)
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_training_curves(history, save_dir)
    
    # 在测试集上评估
    print("\n在测试集上进行最终评估...")
    
    best_model_path = os.path.join(save_dir, 'best_model')
    if os.path.exists(best_model_path):
        model = Wav2Vec2ForCTC.from_pretrained(best_model_path)
        model.to(device)
        
        test_loss, test_wer, test_cer, test_preds, test_refs = evaluate_model(
            model, test_dataloader, processor, device
        )
        
        print(f"\n测试集结果:")
        print(f"Loss: {test_loss:.6f}")
        print(f"WER: {test_wer:.4f}")
        print(f"CER: {test_cer:.4f}")
        
        # 保存测试结果
        test_results = {
            'test_loss': test_loss,
            'test_wer': test_wer,
            'test_cer': test_cer,
            'best_val_wer': min(history['epoch_val_wer']) if history['epoch_val_wer'] else float('inf'),
            'best_val_cer': min(history['epoch_val_cer']) if history['epoch_val_cer'] else float('inf'),
            'total_epochs': len(history['epoch_train_loss'])
        }
        
        with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # 保存预测示例
        with open(os.path.join(save_dir, 'test_predictions.txt'), 'w', encoding='utf-8') as f:
            f.write("测试集预测示例\n")
            f.write("=" * 80 + "\n\n")
            for i in range(min(20, len(test_preds))):
                f.write(f"样本 {i+1}:\n")
                f.write(f"参考: {test_refs[i]}\n")
                f.write(f"预测: {test_preds[i]}\n")
                f.write("-" * 80 + "\n")
    else:
        print("未找到最佳模型，跳过测试集评估")
    
    print(f"\n训练完成！所有结果保存在: {save_dir}")

if __name__ == "__main__":
    main()