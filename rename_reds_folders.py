#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新编码REDS数据集的子文件夹名称
将datasets/REDS/train/HR文件夹下的子文件夹重命名为000, 001, 002...格式
"""

import os
import shutil
from pathlib import Path

def rename_reds_folders(base_path="dataset/REDS/train/HR"):
    """
    重命名REDS数据集子文件夹
    
    Args:
        base_path (str): HR文件夹的路径
    """
    
    # 检查路径是否存在
    if not os.path.exists(base_path):
        print(f"错误: 路径 {base_path} 不存在！")
        return
    
    # 获取所有子文件夹
    folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            folders.append(item)
    
    # 按字母顺序排序
    folders.sort()
    
    print(f"找到 {len(folders)} 个子文件夹")
    print("原始文件夹名称:")
    for i, folder in enumerate(folders):
        print(f"  {i:03d}: {folder}")
    
    # 询问用户确认
    response = input("\n是否继续重命名？(y/N): ")
    if response.lower() not in ['y', 'yes', '是', '确定']:
        print("操作已取消")
        return
    
    # 创建临时目录来避免命名冲突
    temp_dir = os.path.join(os.path.dirname(base_path), "temp_rename")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        # 第一步：将所有文件夹移动到临时目录
        print("\n第一步：移动文件夹到临时目录...")
        for folder in folders:
            old_path = os.path.join(base_path, folder)
            temp_path = os.path.join(temp_dir, folder)
            shutil.move(old_path, temp_path)
            print(f"  移动: {folder} -> temp/{folder}")
        
        # 第二步：用新名称从临时目录移回
        print("\n第二步：用新名称移回...")
        for i, folder in enumerate(folders):
            temp_path = os.path.join(temp_dir, folder)
            new_name = f"{i:03d}"
            new_path = os.path.join(base_path, new_name)
            shutil.move(temp_path, new_path)
            print(f"  重命名: {folder} -> {new_name}")
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        print(f"\n✅ 成功重命名 {len(folders)} 个文件夹！")
        
        # 显示重命名后的结果
        print("\n重命名后的文件夹:")
        new_folders = sorted([f for f in os.listdir(base_path) 
                             if os.path.isdir(os.path.join(base_path, f))])
        for folder in new_folders:
            print(f"  {folder}")
            
    except Exception as e:
        print(f"❌ 重命名过程中出现错误: {e}")
        # 尝试恢复
        if os.path.exists(temp_dir):
            print("尝试恢复原始文件夹...")
            for folder in os.listdir(temp_dir):
                temp_path = os.path.join(temp_dir, folder)
                original_path = os.path.join(base_path, folder)
                if os.path.exists(temp_path):
                    shutil.move(temp_path, original_path)
            shutil.rmtree(temp_dir)
            print("已恢复原始文件夹结构")

def create_mapping_file(base_path="dataset/REDS/train/HR", output_file="folder_mapping.txt"):
    """
    创建文件夹名称映射文件，记录原始名称和新名称的对应关系
    
    Args:
        base_path (str): HR文件夹的路径
        output_file (str): 输出映射文件的路径
    """
    
    if not os.path.exists(base_path):
        print(f"错误: 路径 {base_path} 不存在！")
        return
    
    # 获取所有子文件夹
    folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            folders.append(item)
    
    # 按字母顺序排序
    folders.sort()
    
    # 写入映射文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# REDS数据集文件夹重命名映射\n")
        f.write("# 格式: 新名称 -> 原始名称\n\n")
        for i, folder in enumerate(folders):
            f.write(f"{i:03d} -> {folder}\n")
    
    print(f"映射文件已保存到: {output_file}")

if __name__ == "__main__":
    print("REDS数据集文件夹重命名工具")
    print("=" * 40)
    
    # 检查默认路径
    default_path = "dataset/REDS/train/HR"
    if os.path.exists(default_path):
        print(f"使用默认路径: {default_path}")
        
        # 询问是否先创建映射文件
        response = input("是否先创建名称映射文件？(y/N): ")
        if response.lower() in ['y', 'yes', '是', '确定']:
            create_mapping_file(default_path)
        
        # 执行重命名
        rename_reds_folders(default_path)
    else:
        print(f"默认路径 {default_path} 不存在")
        custom_path = input("请输入HR文件夹的路径: ")
        if custom_path and os.path.exists(custom_path):
            rename_reds_folders(custom_path)
        else:
            print("路径无效，程序退出") 