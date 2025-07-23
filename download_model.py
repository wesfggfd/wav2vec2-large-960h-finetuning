
#!/usr/bin/env python3
"""
模型下载脚本 - 解决HuggingFace连接问题
"""

import os
import requests
import json
from pathlib import Path
import urllib.request
import shutil

def download_from_mirror():
    """从镜像下载模型文件"""
    
    model_name = "facebook/wav2vec2-large-960h"
    mirror_base = "https://hf-mirror.com"
    local_dir = "./models/wav2vec2-large-960h"
    
    # 创建本地目录
    os.makedirs(local_dir, exist_ok=True)
    
    # 需要下载的文件列表
    files_to_download = [
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json"
    ]
    
    print(f"开始下载模型: {model_name}")
    print(f"本地保存路径: {local_dir}")
    
    success_count = 0
    
    for filename in files_to_download:
        try:
            url = f"{mirror_base}/{model_name}/resolve/main/{filename}"
            local_path = os.path.join(local_dir, filename)
            
            print(f"下载: {filename}")
            
            # 使用requests下载
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            print(f"✓ {filename} 下载成功")
            success_count += 1
            
        except Exception as e:
            print(f"✗ {filename} 下载失败: {e}")
            
            # 尝试备用URL
            try:
                backup_url = f"https://huggingface.co/{model_name}/resolve/main/{filename}"
                print(f"尝试备用URL: {backup_url}")
                
                response = requests.get(backup_url, stream=True)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                
                print(f"✓ {filename} 从备用URL下载成功")
                success_count += 1
                
            except Exception as e2:
                print(f"✗ {filename} 备用URL也失败: {e2}")
    
    print(f"\n下载完成: {success_count}/{len(files_to_download)} 个文件成功")
    
    if success_count == len(files_to_download):
        print("所有文件下载成功！")
        print(f"请在训练脚本中使用: model_name = '{local_dir}'")
        return local_dir
    else:
        print("部分文件下载失败，请检查网络连接")
        return None

def test_model_loading(model_path):
    """测试模型加载"""
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        
        print(f"测试加载模型: {model_path}")
        
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
        
        print("✓ 模型加载测试成功！")
        print(f"词汇表大小: {len(processor.tokenizer.vocab)}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型加载测试失败: {e}")
        return False

def main():
    """主函数"""
    print("Wav2Vec2 模型下载工具")
    print("="*50)
    
    # 下载模型
    local_model_path = download_from_mirror()
    
    if local_model_path:
        # 测试加载
        if test_model_loading(local_model_path):
            print("\n使用说明:")
            print("1. 在训练脚本中修改模型路径:")
            print(f"   model_name = '{local_model_path}'")
            print("2. 或者设置环境变量:")
            print(f"   export MODEL_PATH='{local_model_path}'")
        else:
            print("\n模型下载完成但加载失败，请检查文件完整性")
    else:
        print("\n模型下载失败，请尝试以下方法:")
        print("1. 使用VPN或代理")
        print("2. 手动下载文件")
        print("3. 使用huggingface-cli工具")

if __name__ == "__main__":
    main()