# python main.py --template Self_Blind_VSR_Realistic
import os
import jittor as jt
import jittor.nn as nn
import time
import platform
import warnings

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
# Jittor缓存目录
jittor_cache_dir = os.path.join(temp_dir, 'jittor_cache')
if not os.path.exists(jittor_cache_dir):
    os.makedirs(jittor_cache_dir)
os.environ['JITTOR_CACHE_PATH'] = jittor_cache_dir

print(f"⚠️ 临时目录已全部重设为: {temp_dir}")
print(f"   - TEMP = {os.environ['TEMP']}")
print(f"   - TMP = {os.environ['TMP']}")
print(f"   - JITTOR_CACHE_PATH = {os.environ['JITTOR_CACHE_PATH']}")

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
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"系统内存: 总计 {memory.total / (1024**3):.1f} GB, "
              f"可用 {memory.available / (1024**3):.1f} GB")
    except ImportError:
        print("系统内存: 无法获取（psutil未安装）")
    
    # Jittor和CUDA信息
    print(f"Jittor版本: {jt.__version__}")
    if jt.has_cuda:
        print(f"CUDA可用: 是")
        print(f"可用GPU数量: {jt.get_device_count()}")
        try:
            for i in range(jt.get_device_count()):
                print(f"  GPU {i}: 设备可用")
        except:
            print("  GPU信息获取失败")
    else:
        print("CUDA可用: 否 (仅CPU模式可用)")
    print("====================================\n")

if __name__ == '__main__':
    # 打印系统信息
    try:
        print_system_info()
    except Exception as e:
        print(f"无法显示完整系统信息: {e}")

    start_time = time.time()
    
    # 正常执行训练流程
    args = option.args
    
    # 强制设置GPU使用
    if not args.cpu:
        try:
            if jt.has_cuda:
                jt.flags.use_cuda = 1
                # 强制初始化CUDA
                test_tensor = jt.array([1.0])  # 创建一个测试张量来触发CUDA初始化
                print("✅ 使用GPU进行训练")
                print(f"✅ 可用GPU数量: {jt.get_device_count()}")
                print(f"✅ CUDA已启用: {jt.flags.use_cuda}")
                del test_tensor  # 清理测试张量
            else:
                print("❌ 系统未检测到CUDA支持，使用CPU训练")
                jt.flags.use_cuda = 0
                args.cpu = True
        except Exception as e:
            print(f"❌ GPU初始化失败: {e}")
            print("❌ 回退到CPU训练")
            jt.flags.use_cuda = 0
            args.cpu = True
    else:
        jt.flags.use_cuda = 0
        print("🔧 手动设置使用CPU进行训练")
    
    jt.set_global_seed(args.seed)
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