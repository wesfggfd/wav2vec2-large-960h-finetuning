import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
BASE_PATH = "/root/fssd/ASR_task"
DATA_PATH = os.path.join(BASE_PATH, "huggingface/datasets/LibriSpeech")
SAVE_PATH = os.path.join(BASE_PATH, ".cache")
MODELS_PATH = os.path.join(BASE_PATH, ".cache/models")



def download_model_if_needed(model_name, local_path, model_class, processor_class=None):
    """自动下载模型如果需要"""
    # 确保local_path在MODELS_PATH下

    if not local_path.startswith(MODELS_PATH):
        model_folder_name = os.path.basename(local_path)
        local_path = os.path.join(MODELS_PATH, model_folder_name)
    
    if os.path.exists(local_path) and len(os.listdir(local_path)) > 0:
        print(f"模型已存在: {local_path}")
        # 验证模型文件是否完整
        model_files = os.listdir(local_path)
        if any('model' in f for f in model_files) and any('config' in f for f in model_files):
            print(f"模型文件验证通过")
            return True
        else:
            print(f"模型文件不完整，重新下载...")
    
    print(f"下载模型: {model_name}")
    print(f"目标路径: {local_path}")
    print(f"模型缓存目录: {MODELS_PATH}")
    
    # 确保目录存在
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(local_path, exist_ok=True)
    
    try:
        # 下载模型 - 使用HF镜像
        print("正在从HuggingFace下载模型...")
        model = model_class.from_pretrained(
            model_name, 
            cache_dir=MODELS_PATH,
            resume_download=True,  # 支持断点续传
            force_download=False   # 不强制重新下载
        )
        
        # 保存模型到指定路径
        print(f"保存模型到: {local_path}")
        model.save_pretrained(local_path)
        
        # 下载处理器
        if processor_class:
            print("下载处理器...")
            processor = processor_class.from_pretrained(
                model_name,  # 使用默认的Whisper模型
                cache_dir=MODELS_PATH,
                resume_download=True,
                force_download=False
            )
            processor.save_pretrained(local_path)
        
        # 验证下载的文件
        downloaded_files = os.listdir(local_path)
        print(f"下载的文件: {downloaded_files}")
        print(f"模型下载完成: {local_path}")
        
        # 清理临时缓存（可选）
        # 注意：这会删除HuggingFace的缓存，可能影响其他下载
        # import shutil
        # hf_cache = os.path.join(MODELS_PATH, "hub")
        # if os.path.exists(hf_cache):
        #     shutil.rmtree(hf_cache)
        
        return True
    except Exception as e:
        print(f"下载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
if __name__ == "__main__":
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    
    # 示例：下载Whisper模型和处理器
    model_name = "openai/whisper-large-v3"
    local_path = os.path.join(MODELS_PATH, "whisper-large-v3")
    
    success = download_model_if_needed(
        model_name, 
        local_path, 
        WhisperForConditionalGeneration, 
        WhisperProcessor
    )
    
    if success:
        print("模型下载成功！")
    else:
        print("模型下载失败。")