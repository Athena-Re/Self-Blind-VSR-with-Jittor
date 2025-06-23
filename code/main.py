# python main.py --template Self_Blind_VSR_Realistic
import torch
import multiprocessing
import warnings

import data
import model
import loss
import option
from trainer.trainer_flow_video import Trainer_Flow_Video
from logger import logger

warnings.filterwarnings("ignore")

# 添加Windows多进程支持
if __name__ == '__main__':
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

    if args.task == 'FlowVideoSR':
        print("Selected task: {}".format(args.task))
        model = model.Model(args, chkp)
        loss = loss.Loss(args, chkp) if not args.test_only else None
        loader = data.Data(args)
        t = Trainer_Flow_Video(args, loader, model, loss, chkp)
        while not t.terminate():
            t.train()
            t.test()
    else:
        raise NotImplementedError('Task [{:s}] is not found'.format(args.task))

    chkp.done()
