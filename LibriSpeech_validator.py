#!/usr/bin/env python3
"""
LibriSpeech 数据格式验证脚本
快速检查和修复数据问题
"""

import os
import glob
import torchaudio

def check_librispeech_structure(base_path):
    """检查 LibriSpeech 数据结构"""
    print("=" * 60)
    print("LibriSpeech 数据结构验证")
    print("=" * 60)
    
    data_path = os.path.join(base_path, "huggingface/datasets/LibriSpeech")
    print(f"检查路径: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        return False
    
    # 检查子目录
    subdirs = ['train-clean-100', 'train-clean-360', 'dev-clean', 'dev-other']
    found_dirs = []
    
    for subdir in subdirs:
        subdir_path = os.path.join(data_path, subdir)
        if os.path.exists(subdir_path):
            found_dirs.append(subdir)
            print(f"✅ 找到: {subdir}")
            
            # 详细检查这个目录
            check_subdir_structure(subdir_path, subdir)
        else:
            print(f"❌ 缺失: {subdir}")
    
    return len(found_dirs) > 0

def check_subdir_structure(subdir_path, subdir_name):
    """检查子目录结构"""
    print(f"\n📂 检查 {subdir_name} 详细结构:")
    
    # LibriSpeech 标准结构: speaker_id/chapter_id/files
    speaker_dirs = [d for d in os.listdir(subdir_path) 
                   if os.path.isdir(os.path.join(subdir_path, d)) and d.isdigit()]
    
    print(f"   说话人目录数量: {len(speaker_dirs)}")
    
    if not speaker_dirs:
        print("   ❌ 没有找到说话人目录（应该是数字命名的目录）")
        return False
    
    # 检查第一个说话人目录
    first_speaker = speaker_dirs[0]
    speaker_path = os.path.join(subdir_path, first_speaker)
    
    chapter_dirs = [d for d in os.listdir(speaker_path) 
                   if os.path.isdir(os.path.join(speaker_path, d)) and d.isdigit()]
    
    print(f"   第一个说话人 {first_speaker} 的章节数: {len(chapter_dirs)}")
    
    if not chapter_dirs:
        print("   ❌ 没有找到章节目录")
        return False
    
    # 检查第一个章节目录
    first_chapter = chapter_dirs[0]
    chapter_path = os.path.join(speaker_path, first_chapter)
    
    files = os.listdir(chapter_path)
    flac_files = [f for f in files if f.endswith('.flac')]
    trans_files = [f for f in files if f.endswith('.trans.txt')]
    
    print(f"   章节 {first_speaker}/{first_chapter} 内容:")
    print(f"      .flac 文件: {len(flac_files)}")
    print(f"      .trans.txt 文件: {len(trans_files)}")
    
    if flac_files:
        print(f"      第一个音频文件: {flac_files[0]}")
        # 验证文件格式
        first_flac = os.path.join(chapter_path, flac_files[0])
        try:
            info = torchaudio.info(first_flac)
            print(f"      音频信息: {info.sample_rate}Hz, {info.num_frames/info.sample_rate:.2f}s")
        except Exception as e:
            print(f"      ❌ 音频文件损坏: {e}")
    
    if trans_files:
        print(f"      转录文件: {trans_files[0]}")
        # 检查转录文件内容
        trans_file = os.path.join(chapter_path, trans_files[0])
        try:
            with open(trans_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:3]
            print(f"      转录行数: {len(lines)}")
            for i, line in enumerate(lines):
                line = line.strip()
                print(f"      第{i+1}行: {line}")
                
                # 验证对应的音频文件
                parts = line.split(' ', 1)
                if len(parts) >= 2:
                    audio_id = parts[0]
                    expected_audio = os.path.join(chapter_path, f"{audio_id}.flac")
                    exists = os.path.exists(expected_audio)
                    print(f"           对应音频 {audio_id}.flac: {'✅' if exists else '❌'}")
        except Exception as e:
            print(f"      ❌ 读取转录文件失败: {e}")
    
    return len(flac_files) > 0 and len(trans_files) > 0

def create_test_librispeech_data(base_path):
    """创建标准格式的测试数据"""
    print("\n" + "=" * 60)
    print("创建测试 LibriSpeech 数据")
    print("=" * 60)
    
    data_path = os.path.join(base_path, "huggingface/datasets/LibriSpeech")
    
    # 创建标准 LibriSpeech 结构
    test_configs = [
        ("train-clean-100", "103", "1240", [
            ("103-1240-0000", "CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED"),
            ("103-1240-0001", "MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN"),
            ("103-1240-0002", "INTO A LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS"),
        ]),
        ("dev-clean", "84", "121123", [
            ("84-121123-0000", "GO DO THAT GOOD DEED FIRST WHATEVER IT MAY BE"),
            ("84-121123-0001", "THEN YOU WILL HAVE A QUIET RESTING PLACE FOR YOUR CONSCIENCE"),
        ])
    ]
    
    for subset, speaker_id, chapter_id, utterances in test_configs:
        chapter_path = os.path.join(data_path, subset, speaker_id, chapter_id)
        os.makedirs(chapter_path, exist_ok=True)
        
        print(f"创建 {subset}/{speaker_id}/{chapter_id}")
        
        # 创建转录文件
        trans_file = os.path.join(chapter_path, f"{speaker_id}-{chapter_id}.trans.txt")
        with open(trans_file, 'w', encoding='utf-8') as f:
            for audio_id, text in utterances:
                f.write(f"{audio_id} {text}\n")
        
        print(f"   ✅ 转录文件: {os.path.basename(trans_file)}")
        
        # 创建对应的音频文件（空文件，仅用于测试）
        for audio_id, text in utterances:
            audio_file = os.path.join(chapter_path, f"{audio_id}.flac")
            
            # 创建一个最小的有效FLAC文件
            # 这里创建一个空白音频文件，实际应用中应该是真实的音频数据
            create_dummy_flac(audio_file)
            print(f"   ✅ 音频文件: {audio_id}.flac")
    
    print("\n🎉 测试数据创建完成！")
    return True

def create_dummy_flac(output_path):
    """创建一个虚拟的FLAC文件用于测试"""
    try:
        import torch
        import torchaudio
        
        # 创建1秒的静音音频
        sample_rate = 16000
        duration = 1.0  # 1秒
        samples = int(sample_rate * duration)
        
        # 生成非常小的随机噪声（避免完全静音）
        waveform = torch.randn(1, samples) * 0.001
        
        # 保存为FLAC文件
        torchaudio.save(output_path, waveform, sample_rate, format="flac")
        
    except Exception as e:
        # 如果torchaudio保存失败，创建一个假的文件
        with open(output_path, 'wb') as f:
            # 写入FLAC文件头
            f.write(b'fLaC\x00\x00\x00\x22\x10\x00\x10\x00\x00\x00\x0f\x00\x00\x3e\x80\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')

def fix_permissions(base_path):
    """修复文件权限"""
    print("\n修复文件权限...")
    data_path = os.path.join(base_path, "huggingface/datasets/LibriSpeech")
    
    if os.path.exists(data_path):
        try:
            # 递归设置目录权限
            for root, dirs, files in os.walk(data_path):
                # 设置目录权限
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    os.chmod(dir_path, 0o755)
                
                # 设置文件权限
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    os.chmod(file_path, 0o644)
            
            print("✅ 权限修复完成")
        except Exception as e:
            print(f"⚠️ 权限修复失败: {e}")

def main():
    """主函数"""
    base_path = "/root/fssd/ASR_task"
    
    print("LibriSpeech 数据验证和修复工具")
    
    # 1. 检查现有结构
    has_data = check_librispeech_structure(base_path)
    
    if not has_data:
        print(f"\n❌ 没有找到有效的 LibriSpeech 数据")
        
        # 询问是否创建测试数据
        response = input("\n是否创建测试数据？(y/n): ")
        if response.lower() == 'y':
            create_test_librispeech_data(base_path)
            
            # 修复权限
            fix_permissions(base_path)
            
            # 重新验证
            print(f"\n重新验证数据...")
            has_data = check_librispeech_structure(base_path)
        else:
            print("\n请手动准备 LibriSpeech 数据")
            return
    
    if has_data:
        print(f"\n🎉 数据验证通过！可以开始训练了")
        print(f"\n建议在训练脚本中:")
        print(f"1. 使用提供的增强版 LibriSpeechDataset 类")
        print(f"2. 设置 force_reload = True 来强制重新加载数据")
        print(f"3. 检查详细的调试输出")
    else:
        print(f"\n❌ 数据验证失败")

def quick_test():
    """快速测试当前数据"""
    base_path = "/root/fssd/ASR_task"
    data_path = os.path.join(base_path, "huggingface/datasets/LibriSpeech")
    
    print("快速数据测试...")
    
    # 查找任何.flac和.trans.txt文件
    flac_files = glob.glob(os.path.join(data_path, "**/*.flac"), recursive=True)
    trans_files = glob.glob(os.path.join(data_path, "**/*.trans.txt"), recursive=True)
    
    print(f"找到 {len(flac_files)} 个 .flac 文件")
    print(f"找到 {len(trans_files)} 个 .trans.txt 文件")
    
    if flac_files:
        print(f"第一个音频文件: {flac_files[0]}")
    if trans_files:
        print(f"第一个转录文件: {trans_files[0]}")
        
        # 检查第一个转录文件
        try:
            with open(trans_files[0], 'r') as f:
                first_line = f.readline().strip()
            print(f"转录示例: {first_line}")
        except Exception as e:
            print(f"读取转录文件失败: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        main()