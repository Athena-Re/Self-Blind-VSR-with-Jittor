import torch
import jittor as jt
import os
import argparse
from collections import OrderedDict


def convert_pytorch_to_jittor(pytorch_model_path, jittor_model_path):
    """
    将PyTorch模型权重转换为Jittor格式
    """
    print(f"正在加载PyTorch模型: {pytorch_model_path}")
    
    # 加载PyTorch模型权重
    pytorch_state_dict = torch.load(pytorch_model_path, map_location='cpu')
    
    # 创建新的状态字典，转换为Jittor格式
    jittor_state_dict = OrderedDict()
    
    for key, value in pytorch_state_dict.items():
        # 将PyTorch tensor转换为numpy，然后转换为Jittor tensor
        if isinstance(value, torch.Tensor):
            jittor_value = jt.array(value.detach().cpu().numpy())
            jittor_state_dict[key] = jittor_value
        else:
            jittor_state_dict[key] = value
    
    # 保存Jittor模型
    print(f"正在保存Jittor模型: {jittor_model_path}")
    jt.save(jittor_state_dict, jittor_model_path)
    print("转换完成!")
    
    return jittor_state_dict


def main():
    # 直接在代码中指定模型路径
    models_to_convert = [
        {
            'pytorch_model': '../pretrain_models/self_blind_vsr_gaussian.pt',
            'jittor_model': '../pretrain_models/self_blind_vsr_gaussian_jittor.pkl'
        },
        {
            'pytorch_model': '../pretrain_models/self_blind_vsr_realistic.pt', 
            'jittor_model': '../pretrain_models/self_blind_vsr_realistic_jittor.pkl'
        }
    ]
    
    print("开始转换PyTorch模型为Jittor格式...")
    
    for i, model_info in enumerate(models_to_convert, 1):
        pytorch_path = model_info['pytorch_model']
        jittor_path = model_info['jittor_model']
        
        print(f"\n[{i}/{len(models_to_convert)}] 转换模型:")
        print(f"  源文件: {pytorch_path}")
        print(f"  目标文件: {jittor_path}")
        
        if not os.path.exists(pytorch_path):
            print(f"  ❌ 错误: PyTorch模型文件不存在: {pytorch_path}")
            continue
        
        if os.path.exists(jittor_path):
            print(f"  ⚠️  目标文件已存在，跳过: {jittor_path}")
            continue
        
        # 创建输出目录
        output_dir = os.path.dirname(jittor_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            # 执行转换
            convert_pytorch_to_jittor(pytorch_path, jittor_path)
            print(f"  ✅ 转换成功!")
        except Exception as e:
            print(f"  ❌ 转换失败: {str(e)}")
    
    print("\n所有转换任务完成!")


if __name__ == '__main__':
    main() 