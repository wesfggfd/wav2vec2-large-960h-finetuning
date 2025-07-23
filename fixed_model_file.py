#!/usr/bin/env python3
"""
修复模型文件脚本 - 检查和修复下载的模型文件
"""

import os
import json
import requests
import shutil
from pathlib import Path

def check_json_file(file_path):
    """检查JSON文件是否有效"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查文件是否为空
        if not content.strip():
            print(f"❌ {file_path} 文件为空")
            return False
            
        # 检查是否为HTML错误页面
        if content.strip().startswith('<'):
            print(f"❌ {file_path} 包含HTML内容而不是JSON")
            return False
            
        # 尝试解析JSON
        json.loads(content)
        print(f"✅ {file_path} JSON格式正确")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ {file_path} JSON格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path} 检查失败: {e}")
        return False

def download_file_with_retry(url, local_path, max_retries=3):
    """下载文件，支持重试"""
    for attempt in range(max_retries):
        try:
            print(f"下载 {os.path.basename(local_path)} (尝试 {attempt + 1}/{max_retries})")
            
            # 设置请求头
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # 检查响应内容
            if response.headers.get('content-type', '').startswith('text/html'):
                print(f"❌ 获取到HTML页面而不是文件内容")
                continue
                
            # 保存文件
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            # 验证文件大小
            if os.path.getsize(local_path) < 10:
                print(f"❌ 文件太小，可能下载失败")
                continue
                
            print(f"✅ {os.path.basename(local_path)} 下载成功")
            return True
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            if attempt < max_retries - 1:
                print("重试中...")
                continue
    
    return False

def fix_model_files():
    """修复模型文件"""
    
    model_dir = "./models/wav2vec2-large-960h"
    model_name = "facebook/wav2vec2-large-960h"
    
    # 镜像URLs
    mirror_urls = [
        "https://hf-mirror.com",
        "https://huggingface.co"
    ]
    
    # 需要检查的文件
    files_to_check = {
        "config.json": "configuration file",
        "preprocessor_config.json": "preprocessor configuration",
        "tokenizer_config.json": "tokenizer configuration",
        "special_tokens_map.json": "special tokens mapping",
        "vocab.json": "vocabulary file"
    }
    
    print("检查模型文件...")
    print("=" * 50)
    
    # 检查目录是否存在
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        return False
    
    # 检查每个文件
    corrupted_files = []
    
    for filename, description in files_to_check.items():
        file_path = os.path.join(model_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"❌ {filename} 不存在")
            corrupted_files.append(filename)
        elif not check_json_file(file_path):
            print(f"❌ {filename} 损坏")
            corrupted_files.append(filename)
    
    # 如果有损坏的文件，重新下载
    if corrupted_files:
        print(f"\n发现 {len(corrupted_files)} 个损坏的文件，重新下载...")
        
        for filename in corrupted_files:
            success = False
            
            for mirror_url in mirror_urls:
                url = f"{mirror_url}/{model_name}/resolve/main/{filename}"
                file_path = os.path.join(model_dir, filename)
                
                print(f"\n尝试从 {mirror_url} 下载 {filename}")
                
                if download_file_with_retry(url, file_path):
                    # 验证下载的文件
                    if check_json_file(file_path):
                        success = True
                        break
                    else:
                        print(f"❌ 下载的文件仍然损坏")
            
            if not success:
                print(f"❌ 无法修复 {filename}")
                return False
    
    print(f"\n✅ 所有文件检查完成")
    return True

def create_correct_config_files():
    """创建正确的配置文件（如果下载失败）"""
    
    model_dir = "./models/wav2vec2-large-960h"
    
    # 正确的配置文件内容
    config_content = {
        "_name_or_path": "facebook/wav2vec2-large-960h",
        "activation_dropout": 0.1,
        "apply_spec_augment": True,
        "architectures": ["Wav2Vec2ForCTC"],
        "attention_dropout": 0.1,
        "bos_token_id": 1,
        "classifier_proj_size": 256,
        "codevector_dim": 768,
        "contrastive_logits_temperature": 0.1,
        "conv_bias": True,
        "conv_dim": [512, 512, 512, 512, 512, 512, 512],
        "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
        "conv_stride": [5, 2, 2, 2, 2, 2, 2],
        "ctc_loss_reduction": "mean",
        "ctc_zero_infinity": False,
        "diversity_loss_weight": 0.1,
        "do_stable_layer_norm": True,
        "eos_token_id": 2,
        "feat_extract_activation": "gelu",
        "feat_extract_dropout": 0.0,
        "feat_extract_norm": "layer",
        "feat_proj_dropout": 0.1,
        "feat_quantizer_dropout": 0.0,
        "final_dropout": 0.0,
        "freeze_feat_extract_train": True,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "layer_norm_eps": 1e-05,
        "layerdrop": 0.1,
        "mask_feature_length": 10,
        "mask_feature_min_masks": 0,
        "mask_feature_prob": 0.0,
        "mask_time_length": 10,
        "mask_time_min_masks": 2,
        "mask_time_prob": 0.05,
        "model_type": "wav2vec2",
        "num_adapter_layers": 3,
        "num_attention_heads": 16,
        "num_codevector_groups": 2,
        "num_codevectors_per_group": 320,
        "num_conv_pos_embedding_groups": 16,
        "num_conv_pos_embeddings": 128,
        "num_feat_extract_layers": 7,
        "num_hidden_layers": 24,
        "num_negatives": 100,
        "pad_token_id": 0,
        "proj_codevector_dim": 768,
        "quantize_targets": True,
        "sample_rate": 16000,
        "target_feature_length": 1024,
        "tokenizer_class": "Wav2Vec2CTCTokenizer",
        "torch_dtype": "float32",
        "transformers_version": "4.21.0",
        "use_weighted_layer_sum": False,
        "vocab_size": 32
    }
    
    preprocessor_config_content = {
        "do_normalize": True,
        "feature_extractor_type": "Wav2Vec2FeatureExtractor",
        "feature_size": 1,
        "padding_side": "right",
        "padding_value": 0.0,
        "processor_class": "Wav2Vec2Processor",
        "return_attention_mask": True,
        "sampling_rate": 16000,
        "tokenizer_class": "Wav2Vec2CTCTokenizer"
    }
    
    tokenizer_config_content = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": "<pad>",
        "processor_class": "Wav2Vec2Processor",
        "tokenizer_class": "Wav2Vec2CTCTokenizer",
        "unk_token": "<unk>",
        "word_delimiter_token": "|"
    }
    
    special_tokens_map_content = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>"
    }
    
    vocab_content = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "|": 4,
        "E": 5,
        "T": 6,
        "A": 7,
        "O": 8,
        "N": 9,
        "I": 10,
        "H": 11,
        "S": 12,
        "R": 13,
        "D": 14,
        "L": 15,
        "U": 16,
        "M": 17,
        "W": 18,
        "C": 19,
        "F": 20,
        "G": 21,
        "Y": 22,
        "P": 23,
        "B": 24,
        "V": 25,
        "K": 26,
        "J": 27,
        "X": 28,
        "Q": 29,
        "Z": 30,
        "'": 31
    }
    
    # 保存配置文件
    configs = {
        "config.json": config_content,
        "preprocessor_config.json": preprocessor_config_content,
        "tokenizer_config.json": tokenizer_config_content,
        "special_tokens_map.json": special_tokens_map_content,
        "vocab.json": vocab_content
    }
    
    print("创建正确的配置文件...")
    for filename, content in configs.items():
        file_path = os.path.join(model_dir, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
            print(f"✅ {filename} 创建成功")
        except Exception as e:
            print(f"❌ {filename} 创建失败: {e}")
            return False
    
    return True

def test_model_loading():
    """测试模型加载"""
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        
        model_path = "./models/wav2vec2-large-960h"
        print(f"测试加载模型: {model_path}")
        
        # 测试处理器加载
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        print("✅ 处理器加载成功")
        
        # 测试模型加载（只加载配置，不加载权重）
        from transformers import Wav2Vec2Config
        config = Wav2Vec2Config.from_pretrained(model_path)
        print("✅ 配置加载成功")
        
        # 如果pytorch_model.bin存在，测试完整模型加载
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            model = Wav2Vec2ForCTC.from_pretrained(model_path)
            print("✅ 完整模型加载成功")
            print(f"词汇表大小: {len(processor.tokenizer.vocab)}")
            print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        else:
            print("⚠️ pytorch_model.bin 不存在，只能加载配置")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        return False

def main():
    """主函数"""
    print("Wav2Vec2 模型文件修复工具")
    print("=" * 50)
    
    # 检查并修复文件
    if fix_model_files():
        print("\n文件修复完成，测试模型加载...")
        
        if test_model_loading():
            print("\n✅ 模型修复成功！可以正常使用了")
        else:
            print("\n配置文件可能仍有问题，尝试创建标准配置文件...")
            if create_correct_config_files():
                print("重新测试模型加载...")
                if test_model_loading():
                    print("\n✅ 模型修复成功！")
                else:
                    print("\n❌ 仍然无法加载模型")
            else:
                print("\n❌ 无法创建配置文件")
    else:
        print("\n❌ 文件修复失败")

if __name__ == "__main__":
    main()