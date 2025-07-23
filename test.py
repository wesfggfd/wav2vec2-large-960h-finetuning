# GPU利用率监控完整解决方案
import os
import sys
import subprocess
import time
import torch

class GPUUtilizationMonitor:
    def __init__(self):
        self.nvml_available = False
        self.nvidia_smi_available = False
        self.setup_monitoring()
    
    def setup_monitoring(self):
        """设置GPU监控方法"""
        # 方法1: 尝试pynvml
        if self._try_install_and_import_pynvml():
            self.nvml_available = True
            print("✓ pynvml 可用")
        
        # 方法2: 检查nvidia-smi
        if self._check_nvidia_smi():
            self.nvidia_smi_available = True
            print("✓ nvidia-smi 可用")
        
        if not (self.nvml_available or self.nvidia_smi_available):
            print("❌ 无可用的GPU监控方法")
    
    def _try_install_and_import_pynvml(self):
        """尝试安装并导入pynvml"""
        try:
            import pynvml
            return True
        except ImportError:
            print("正在尝试安装pynvml...")
            try:
                # 尝试安装pynvml
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pynvml"])
                import pynvml
                return True
            except:
                try:
                    # 尝试安装nvidia-ml-py3
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "nvidia-ml-py3"])
                    import pynvml
                    return True
                except:
                    print("❌ 无法安装pynvml库")
                    return False
    
    def _check_nvidia_smi(self):
        """检查nvidia-smi是否可用"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def get_gpu_utilization_pynvml(self, device_id=None):
        """使用pynvml获取GPU利用率"""
        if not self.nvml_available:
            return None
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            if device_id is None:
                if torch.cuda.is_available():
                    device_id = torch.cuda.current_device()
                else:
                    device_id = 0
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # 获取更多信息
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # 兼容不同pynvml版本的GPU名称获取
            raw_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(raw_name, bytes):
                name = raw_name.decode('utf-8')
            else:
                name = str(raw_name)
            
            return {
                'gpu_utilization': float(util.gpu),
                'memory_utilization': float(util.memory),
                'memory_used': memory_info.used // (1024**2),  # MB
                'memory_total': memory_info.total // (1024**2),  # MB
                'gpu_name': name,
                'device_id': device_id
            }
        except Exception as e:
            print(f"pynvml方法出错: {e}")
            return None
    
    def get_gpu_utilization_nvidia_smi(self, device_id=None):
        """使用nvidia-smi获取GPU利用率"""
        if not self.nvidia_smi_available:
            return None
        
        try:
            if device_id is not None:
                cmd = ['nvidia-smi', f'--id={device_id}']
            else:
                cmd = ['nvidia-smi']
            
            cmd.extend(['--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,name', 
                       '--format=csv,noheader,nounits'])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_data = []
                
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpu_data.append({
                            'gpu_utilization': float(parts[0]),
                            'memory_utilization': float(parts[1]),
                            'memory_used': int(parts[2]),
                            'memory_total': int(parts[3]),
                            'gpu_name': parts[4],
                            'device_id': i
                        })
                
                if device_id is not None and device_id < len(gpu_data):
                    return gpu_data[device_id]
                elif len(gpu_data) > 0:
                    return gpu_data[0] if device_id is None else gpu_data
                
        except Exception as e:
            print(f"nvidia-smi方法出错: {e}")
            return None
    
    def get_gpu_utilization(self, device_id=None):
        """获取GPU利用率（优先使用pynvml）"""
        # 优先使用pynvml
        result = self.get_gpu_utilization_pynvml(device_id)
        if result is not None:
            return result
        
        # 备用nvidia-smi
        result = self.get_gpu_utilization_nvidia_smi(device_id)
        if result is not None:
            return result
        
        return None
    
    def get_all_gpu_info(self):
        """获取所有GPU信息"""
        if self.nvml_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                all_gpus = []
                for i in range(device_count):
                    gpu_info = self.get_gpu_utilization_pynvml(i)
                    if gpu_info:
                        all_gpus.append(gpu_info)
                return all_gpus
            except:
                pass
        
        # 备用方法
        return self.get_gpu_utilization_nvidia_smi()
    
    def monitor_continuously(self, interval=1, duration=60):
        """连续监控GPU利用率"""
        print(f"开始监控GPU利用率，间隔{interval}秒，持续{duration}秒")
        print("=" * 80)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            gpu_info = self.get_gpu_utilization()
            
            if gpu_info:
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] GPU: {gpu_info['gpu_utilization']:5.1f}% | "
                      f"内存: {gpu_info['memory_utilization']:5.1f}% | "
                      f"显存: {gpu_info['memory_used']:5d}/{gpu_info['memory_total']:5d}MB | "
                      f"设备: {gpu_info['gpu_name']}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 无法获取GPU信息")
            
            time.sleep(interval)

def main():
    """主函数和测试代码"""
    print("GPU利用率监控系统")
    print("=" * 50)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✓ CUDA可用，检测到{torch.cuda.device_count()}个GPU设备")
        print(f"✓ 当前设备: {torch.cuda.get_device_name()}")
    else:
        print("❌ CUDA不可用或没有GPU设备")
    
    # 初始化监控器
    monitor = GPUUtilizationMonitor()
    
    print("\n" + "=" * 50)
    
    # 测试1: 获取单个GPU信息
    print("测试1: 获取当前GPU利用率")
    gpu_info = monitor.get_gpu_utilization()
    if gpu_info:
        print(f"GPU利用率: {gpu_info['gpu_utilization']:.1f}%")
        print(f"内存利用率: {gpu_info['memory_utilization']:.1f}%")
        print(f"显存使用: {gpu_info['memory_used']}/{gpu_info['memory_total']} MB")
        print(f"GPU名称: {gpu_info['gpu_name']}")
    else:
        print("❌ 无法获取GPU信息")
    
    # 测试2: 获取所有GPU信息
    print("\n测试2: 获取所有GPU信息")
    all_gpus = monitor.get_all_gpu_info()
    if all_gpus:
        for i, gpu in enumerate(all_gpus):
            print(f"GPU {i}: {gpu['gpu_name']} - 利用率: {gpu['gpu_utilization']:.1f}%")
    
    # 测试3: 连续监控（可选）
    print("\n是否开始连续监控？(y/n): ", end="")
    try:
        choice = input().lower()
        if choice == 'y':
            monitor.monitor_continuously(interval=2, duration=30)
    except KeyboardInterrupt:
        print("\n监控已停止")

# 诊断函数
def diagnose_gpu_environment():
    """诊断GPU环境"""
    print("GPU环境诊断")
    print("=" * 30)
    
    # 检查CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检查nvidia-smi
    print(f"\nnvidia-smi可用: ", end="")
    try:
        result = subprocess.run(['nvidia-smi', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ ({version_line})")
        else:
            print("❌")
    except:
        print("❌")
    
    # 检查Python包
    packages = ['pynvml', 'nvidia-ml-py3', 'nvidia-ml-py']
    print(f"\nPython包检查:")
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  {pkg}: ✓")
        except ImportError:
            print(f"  {pkg}: ❌")

if __name__ == "__main__":
    # 可以选择运行诊断或主程序
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        diagnose_gpu_environment()
    else:
        main()