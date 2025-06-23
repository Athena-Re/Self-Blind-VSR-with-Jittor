# python main.py --template Self_Blind_VSR_Realistic
import os
import torch
import multiprocessing
import warnings
import time
import platform
import psutil

# 设置临时目录环境变量（确保在导入其他模块前执行）
# 创建临时目录（如果不存在）
temp_dir = 'D:\\TEMP'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
    print(f"✅ 临时目录已创建: {temp_dir}")

# 设置所有可能的临时目录环境变量
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir
os.environ['TMPDIR'] = temp_dir
# CUDA特定的临时目录
os.environ['CUDA_CACHE_PATH'] = os.path.join(temp_dir, 'cuda_cache')
if not os.path.exists(os.environ['CUDA_CACHE_PATH']):
    os.makedirs(os.environ['CUDA_CACHE_PATH'])

# 设置PyTorch的临时目录
torch.hub.set_dir(os.path.join(temp_dir, 'torch_hub'))

print(f"⚠️ 临时目录已全部重设为: {temp_dir}")
print(f"   - TEMP = {os.environ['TEMP']}")
print(f"   - TMP = {os.environ['TMP']}")
print(f"   - CUDA_CACHE_PATH = {os.environ['CUDA_CACHE_PATH']}")

# 尝试使用CUDA实现（已修改临时目录，避免中文路径问题）
os.environ['FORCE_CORRELATION_PYTORCH'] = 'FALSE'
print("✅ 将尝试使用CUDA相关性实现（如编译失败会自动回退到PyTorch实现）")

import data
import model
import loss
import option
from trainer.trainer_flow_video import Trainer_Flow_Video
from logger import logger

warnings.filterwarnings("ignore")

def print_system_info():
    """打印系统信息"""
    print("\n====================================")
    print("系统信息")
    print("====================================")
    print(f"操作系统: {platform.system()} {platform.version()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"处理器: {platform.processor()}")
    
    # 内存信息
    memory = psutil.virtual_memory()
    print(f"系统内存: 总计 {memory.total / (1024**3):.1f} GB, "
          f"可用 {memory.available / (1024**3):.1f} GB")
    
    # PyTorch和CUDA信息
    print(f"PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA可用: 是 (版本 {torch.version.cuda})")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - 可用显存: {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GB")
    else:
        print("CUDA可用: 否 (仅CPU模式可用)")
    print("====================================\n")

# 添加Windows多进程支持
if __name__ == '__main__':
    # 打印系统信息
    try:
        print_system_info()
    except Exception as e:
        print(f"无法显示完整系统信息: {e}")

    start_time = time.time()
    
    # 设置多进程启动方式
    if torch.cuda.is_available():
        # CUDA设备下使用spawn方法
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    # 正常执行训练流程
    args = option.args
    
    # 调整训练设备设置
    if args.cpu:
        print("⚠️ 强制使用CPU模式运行")
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            cuda_idx = 0
            device = torch.device(f'cuda:{cuda_idx}')
            print(f"✅ 使用GPU: {torch.cuda.get_device_name(cuda_idx)}")
            print(f"   - 可用显存: {torch.cuda.get_device_properties(cuda_idx).total_memory/1024**3:.1f} GB")
            print(f"   - CUDA版本: {torch.version.cuda}")
        else:
            device = torch.device('cpu')
            print("⚠️ 未检测到GPU，自动切换到CPU模式")
            args.cpu = True
    
    torch.manual_seed(args.seed)
    chkp = logger.Logger(args)

    print("\n====================================")
    print(f"任务: {args.task}")
    print(f"模板: {args.template}")
    print(f"数据集: {args.data_train} (训练), {args.data_test} (测试)")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"总训练轮次: {args.epochs}")
    print("====================================\n")
    
    try:
        if args.task == 'FlowVideoSR':
            print("🔄 准备开始训练任务: {}".format(args.task))
            model = model.Model(args, chkp)
            print("✅ 模型创建成功")
            
            loss = loss.Loss(args, chkp) if not args.test_only else None
            if not args.test_only:
                print("✅ 损失函数创建成功")
            
            print("🔄 正在加载数据...")
            loader = data.Data(args)
            print("✅ 数据加载成功")
            
            print("🔄 初始化训练器...")
            t = Trainer_Flow_Video(args, loader, model, loss, chkp)
            print("✅ 训练器初始化成功")
            
            train_start_time = time.time()
            while not t.terminate():
                t.train()
                t.test()
                
            total_train_time = time.time() - train_start_time
            print(f"\n====================================")
            print(f"训练完成！")
            print(f"总训练时间: {total_train_time/60:.2f}分钟 ({total_train_time/3600:.2f}小时)")
            print(f"====================================\n")
        else:
            raise NotImplementedError('Task [{:s}] is not found'.format(args.task))
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\n总运行时间: {total_time/60:.2f}分钟 ({total_time/3600:.2f}小时)")
    chkp.done()
