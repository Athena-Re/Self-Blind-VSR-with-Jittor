import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 解决中文乱码问题
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 检查并设置可用的中文字体
def setup_chinese_fonts():
    """设置中文字体，解决乱码问题"""
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            print(f"使用中文字体: {font}")
            break
    else:
        print("警告: 未找到中文字体，可能出现乱码")
        # 在Windows上尝试使用系统字体
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        except:
            pass

# 调用字体设置
setup_chinese_fonts()

sns.set_style("whitegrid")
sns.set_palette("husl")

class LogAnalyzer:
    def __init__(self, log_files):
        self.log_files = log_files
        self.data = {}
        self.output_dir = 'visualization'
        
        # 创建输出目录
        self.create_output_directory()
    
    def create_output_directory(self):
        """创建输出目录"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")
        else:
            print(f"输出目录已存在: {self.output_dir}")
    
    def ensure_chinese_font(self):
        """确保中文字体设置正确"""
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
    def parse_log_file(self, log_path, name):
        """解析单个日志文件"""
        data = {
            'videos': [],
            'frames': [],
            'psnr': [],
            'ssim': [],
            'pre_time': [],
            'forward_time': [],
            'post_time': [],
            'total_time': [],
            'video_stats': {}
        }
        
        print(f"正在解析日志文件: {log_path}")
        
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # 匹配推理结果行
        pattern = r'> (\d+)-(\d+) PSNR=([0-9.]+), SSIM=([0-9.]+) pre_time:([0-9.]+)s, forward_time:([0-9.]+)s, post_time:([0-9.]+)s, total_time:([0-9.]+)s'
        matches = re.findall(pattern, content)
        
        for match in matches:
            video, frame, psnr, ssim, pre_time, forward_time, post_time, total_time = match
            data['videos'].append(video)
            data['frames'].append(int(frame))
            data['psnr'].append(float(psnr))
            data['ssim'].append(float(ssim))
            data['pre_time'].append(float(pre_time))
            data['forward_time'].append(float(forward_time))
            data['post_time'].append(float(post_time))
            data['total_time'].append(float(total_time))
        
        # 计算每个视频的统计信息
        for video in set(data['videos']):
            video_indices = [i for i, v in enumerate(data['videos']) if v == video]
            data['video_stats'][video] = {
                'avg_psnr': np.mean([data['psnr'][i] for i in video_indices]),
                'avg_ssim': np.mean([data['ssim'][i] for i in video_indices]),
                'avg_time': np.mean([data['total_time'][i] for i in video_indices]),
                'total_frames': len(video_indices)
            }
        
        print(f"解析完成: {len(matches)} 个帧结果")
        return data
    
    def load_all_data(self):
        """加载所有日志数据"""
        for name, path in self.log_files.items():
            if os.path.exists(path):
                self.data[name] = self.parse_log_file(path, name)
            else:
                print(f"警告: 文件不存在 {path}")
    
    def create_individual_charts(self):
        """创建多个独立的图表"""
        if not self.data:
            print("没有数据可以可视化")
            return
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A', '#DDA0DD']
        
        # 创建独立图表
        self.create_psnr_comparison_chart(colors)
        self.create_ssim_comparison_chart(colors)
        self.create_time_performance_chart(colors)
        self.create_video_psnr_distribution_chart(colors)
        self.create_video_ssim_distribution_chart(colors)
        self.create_time_breakdown_chart(colors)
        self.create_psnr_trend_chart(colors)
        self.create_performance_radar_chart(colors)
        self.create_efficiency_chart(colors)
        self.create_quality_vs_speed_chart(colors)
        
        print(f"所有图表已生成完成！保存在目录: {self.output_dir}")
        print(f"共生成 10 个独立图表文件")
    
    def create_psnr_comparison_chart(self, colors):
        """创建PSNR对比图表"""
        self.ensure_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_list = []
        labels = []
        
        for i, (name, data) in enumerate(self.data.items()):
            if data['psnr']:
                data_list.append(data['psnr'])
                labels.append(name)
        
        if data_list:
            parts = ax.violinplot(data_list, positions=range(len(data_list)), 
                                showmeans=True, showmedians=True)
            
            # 美化小提琴图
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
            
            # 添加统计信息标注
            for i, data in enumerate(data_list):
                mean_val = np.mean(data)
                ax.text(i, mean_val + 0.5, f'μ={mean_val:.2f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
            ax.set_title('PSNR 分布对比\n不同版本的峰值信噪比性能分析', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'chart_psnr_comparison.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_ssim_comparison_chart(self, colors):
        """创建SSIM对比图表"""
        self.ensure_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_list = []
        labels = []
        
        for i, (name, data) in enumerate(self.data.items()):
            if data['ssim']:
                data_list.append(data['ssim'])
                labels.append(name)
        
        if data_list:
            parts = ax.violinplot(data_list, positions=range(len(data_list)), 
                                showmeans=True, showmedians=True)
            
            # 美化小提琴图
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
            
            # 添加统计信息标注
            for i, data in enumerate(data_list):
                mean_val = np.mean(data)
                ax.text(i, mean_val + 0.01, f'μ={mean_val:.4f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_ylabel('SSIM', fontsize=12, fontweight='bold')
            ax.set_title('SSIM 分布对比\n结构相似性指数性能分析', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'chart_ssim_comparison.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_time_performance_chart(self, colors):
        """创建时间性能对比图表"""
        self.ensure_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(self.data.keys())
        avg_times = []
        
        for name in names:
            if self.data[name]['total_time']:
                avg_times.append(np.mean(self.data[name]['total_time']))
            else:
                avg_times.append(0)
        
        bars = ax.bar(names, avg_times, color=colors[:len(names)], alpha=0.8, 
                     edgecolor='black', linewidth=1)
        
        # 添加数值标签和效率标注
        for i, (bar, time) in enumerate(zip(bars, avg_times)):
            height = bar.get_height()
            fps = 1.0 / time if time > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{time:.3f}s\n({fps:.1f} FPS)', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
            
            # 添加性能等级标注
            if fps > 30:
                level = "优秀"
                color = '#4CAF50'
            elif fps > 15:
                level = "良好"
                color = '#FF9800'
            else:
                level = "一般"
                color = '#F44336'
            
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   level, ha='center', va='center', 
                   fontsize=12, fontweight='bold', color=color)
        
        ax.set_ylabel('平均推理时间 (秒)', fontsize=12, fontweight='bold')
        ax.set_title('推理时间性能对比\n处理速度与效率分析', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(names, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'chart_time_performance.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_video_psnr_distribution_chart(self, colors):
        """创建各视频PSNR分布图表"""
        self.ensure_chinese_font()
        fig, ax = plt.subplots(figsize=(12, 7))
        
        video_names = set()
        for data in self.data.values():
            video_names.update(data['videos'])
        video_names = sorted(list(video_names))
        
        x = np.arange(len(video_names))
        width = 0.35
        
        for i, (name, data) in enumerate(self.data.items()):
            video_psnrs = []
            for video in video_names:
                video_indices = [j for j, v in enumerate(data['videos']) if v == video]
                if video_indices:
                    psnr_val = np.mean([data['psnr'][j] for j in video_indices])
                    video_psnrs.append(psnr_val)
                else:
                    video_psnrs.append(0)
            
            x_pos = x + i * width
            bars = ax.bar(x_pos, video_psnrs, width, label=name, 
                         color=colors[i % len(colors)], alpha=0.8, 
                         edgecolor='black', linewidth=0.5)
            
            # 添加数值标签
            for bar, psnr in zip(bars, video_psnrs):
                if psnr > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{psnr:.2f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
        
        ax.set_xlabel('视频序列', fontsize=12, fontweight='bold')
        ax.set_ylabel('平均PSNR (dB)', fontsize=12, fontweight='bold')
        ax.set_title('各视频序列PSNR性能对比\n不同测试视频的质量重建效果', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(self.data) - 1) / 2)
        ax.set_xticklabels([f'Video {v}' for v in video_names], fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'chart_video_psnr_distribution.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_video_ssim_distribution_chart(self, colors):
        """创建各视频SSIM分布图表"""
        self.ensure_chinese_font()
        fig, ax = plt.subplots(figsize=(12, 7))
        
        video_names = set()
        for data in self.data.values():
            video_names.update(data['videos'])
        video_names = sorted(list(video_names))
        
        x = np.arange(len(video_names))
        width = 0.35
        
        for i, (name, data) in enumerate(self.data.items()):
            video_ssims = []
            for video in video_names:
                video_indices = [j for j, v in enumerate(data['videos']) if v == video]
                if video_indices:
                    ssim_val = np.mean([data['ssim'][j] for j in video_indices])
                    video_ssims.append(ssim_val)
                else:
                    video_ssims.append(0)
            
            x_pos = x + i * width
            bars = ax.bar(x_pos, video_ssims, width, label=name, 
                         color=colors[i % len(colors)], alpha=0.8,
                         edgecolor='black', linewidth=0.5)
            
            # 添加数值标签
            for bar, ssim in zip(bars, video_ssims):
                if ssim > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{ssim:.3f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
        
        ax.set_xlabel('视频序列', fontsize=12, fontweight='bold')
        ax.set_ylabel('平均SSIM', fontsize=12, fontweight='bold')
        ax.set_title('各视频序列SSIM性能对比\n结构相似性在不同测试视频上的表现', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(self.data) - 1) / 2)
        ax.set_xticklabels([f'Video {v}' for v in video_names], fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'chart_video_ssim_distribution.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_time_breakdown_chart(self, colors):
        """创建时间组成分析图表"""
        self.ensure_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(self.data.keys())
        pre_times = []
        forward_times = []
        post_times = []
        
        for name in names:
            data = self.data[name]
            if data['pre_time']:
                pre_times.append(np.mean(data['pre_time']))
                forward_times.append(np.mean(data['forward_time']))
                post_times.append(np.mean(data['post_time']))
            else:
                pre_times.append(0)
                forward_times.append(0)
                post_times.append(0)
        
        x = np.arange(len(names))
        width = 0.6
        
        p1 = ax.bar(x, pre_times, width, label='预处理时间', color=colors[0], alpha=0.8)
        p2 = ax.bar(x, forward_times, width, bottom=pre_times, label='前向推理时间', color=colors[1], alpha=0.8)
        p3 = ax.bar(x, post_times, width, bottom=np.array(pre_times) + np.array(forward_times), 
                   label='后处理时间', color=colors[2], alpha=0.8)
        
        # 添加百分比标注
        for i, (pre, forward, post) in enumerate(zip(pre_times, forward_times, post_times)):
            total = pre + forward + post
            if total > 0:
                # 前向推理时间占比（最重要的指标）
                forward_ratio = forward / total * 100
                ax.text(i, pre + forward/2, f'{forward_ratio:.1f}%', 
                       ha='center', va='center', fontsize=10, fontweight='bold', color='white')
                
                # 总时间标注
                ax.text(i, total + 0.01, f'{total:.3f}s', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('实现版本', fontsize=12, fontweight='bold')
        ax.set_ylabel('时间 (秒)', fontsize=12, fontweight='bold')
        ax.set_title('推理时间组成分析\n各阶段耗时占比与优化方向', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'chart_time_breakdown.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_psnr_trend_chart(self, colors):
        """创建PSNR变化趋势图表"""
        self.ensure_chinese_font()
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for i, (name, data) in enumerate(self.data.items()):
            if data['psnr']:
                # 计算移动平均以平滑曲线
                window_size = min(10, max(1, len(data['psnr']) // 20))
                if window_size > 1:
                    smoothed_psnr = np.convolve(data['psnr'], np.ones(window_size)/window_size, mode='valid')
                    frames = range(window_size-1, len(data['psnr']))
                else:
                    smoothed_psnr = data['psnr']
                    frames = range(len(data['psnr']))
                
                ax.plot(frames, smoothed_psnr, label=f'{name} (均值: {np.mean(data["psnr"]):.2f}dB)', 
                       color=colors[i % len(colors)], linewidth=2.5, alpha=0.8)
                
                # 添加最高点和最低点标注
                max_idx = np.argmax(smoothed_psnr)
                min_idx = np.argmin(smoothed_psnr)
                max_frame = frames[max_idx]
                min_frame = frames[min_idx]
                
                ax.annotate(f'最高: {smoothed_psnr[max_idx]:.2f}dB', 
                           xy=(max_frame, smoothed_psnr[max_idx]), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i % len(colors)], alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color=colors[i % len(colors)]))
        
        ax.set_xlabel('帧序号', fontsize=12, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        ax.set_title('PSNR随推理进程变化趋势\n质量稳定性与波动分析', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'chart_psnr_trend.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_performance_radar_chart(self, colors):
        """创建性能雷达图表"""
        self.ensure_chinese_font()
        
        # 增大图表尺寸，为标题和图例留出更多空间
        fig = plt.figure(figsize=(14, 12))
        
        # 创建子图，为标题预留空间
        ax = fig.add_subplot(111, projection='polar')
        
        categories = ['PSNR\n(图像质量)', 'SSIM\n(结构相似)', '推理速度\n(FPS)', '质量稳定性\n(一致性)']
        N = len(categories)
        
        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 完成圆形
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # 绘制每个版本的雷达图
        for i, (name, data) in enumerate(self.data.items()):
            if data['psnr']:
                # 归一化指标 (0-1)
                avg_psnr = np.mean(data['psnr'])
                avg_ssim = np.mean(data['ssim'])
                avg_time = np.mean(data['total_time'])
                psnr_std = np.std(data['psnr'])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                psnr_norm = min(1.0, max(0.0, (avg_psnr - 20) / 15))  # PSNR范围20-35
                ssim_norm = avg_ssim  # SSIM本身就是0-1
                speed_norm = min(1.0, max(0.0, min(1.0, fps / 30)))  # 速度归一化到30FPS基准
                stability_norm = min(1.0, max(0.0, 1 - (psnr_std / max(avg_psnr, 1))))  # 稳定性
                
                values = [psnr_norm, ssim_norm, speed_norm, stability_norm]
                values += values[:1]  # 完成圆形
                
                # 绘制雷达图线条和填充
                ax.plot(angles, values, 'o-', linewidth=4, label=name, 
                       color=colors[i % len(colors)], alpha=0.9, markersize=10)
                ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
                
                # 添加数值标注 - 调整位置和字体
                for j, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
                    if value > 0.3:  # 显示更多数值
                        # 根据角度调整标注位置
                        radius_offset = 0.12 if value > 0.8 else 0.08
                        ax.text(angle, value + radius_offset, f'{value:.2f}', 
                               ha='center', va='center', fontsize=11, fontweight='bold',
                               color=colors[i % len(colors)],
                               bbox=dict(boxstyle='round,pad=0.2', 
                                       facecolor='white', alpha=0.8, edgecolor=colors[i % len(colors)]))
        
        # 优化标签和网格线
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=13, fontweight='bold', ha='center')
        ax.set_ylim(0, 1.2)  # 增加上边界，为标注留空间
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=11, alpha=0.7)
        
        # 优化网格显示
        ax.grid(True, alpha=0.4, linewidth=1)
        ax.set_facecolor('#fafafa')
        
        # 设置标题在右上角
        ax.text(0.95, 0.95, '综合性能雷达图', 
               transform=ax.transAxes, ha='right', va='top',
               fontsize=18, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # 添加副标题在右上角下方
        ax.text(0.95, 0.88, '多维度性能评估与对比分析', 
               transform=ax.transAxes, ha='right', va='top',
               fontsize=12, style='italic', alpha=0.8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
        
        # 优化图例位置和样式
        legend = ax.legend(loc='center', bbox_to_anchor=(0.5, -0.25), 
                          fontsize=13, ncol=len(self.data), frameon=True,
                          fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # 添加性能说明文字
        explanation_text = "评分说明: PSNR (峰值信噪比), SSIM (结构相似性), 推理速度 (FPS), 质量稳定性 (一致性)\n数值越接近外圈代表性能越好"
        ax.text(0.5, -0.35, explanation_text, 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=10, alpha=0.7, style='italic',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
        
        # 调整布局，确保所有元素都能显示
        plt.subplots_adjust(bottom=0.2, top=0.95)
        
        plt.savefig(os.path.join(self.output_dir, 'chart_performance_radar.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
        plt.show()
    
    def create_efficiency_chart(self, colors):
        """创建效率分析图表"""
        self.ensure_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(self.data.keys())
        fps_values = []
        
        for name in names:
            if self.data[name]['total_time']:
                avg_time = np.mean(self.data[name]['total_time'])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                fps_values.append(fps)
            else:
                fps_values.append(0)
        
        bars = ax.bar(names, fps_values, color=colors[:len(names)], alpha=0.8,
                     edgecolor='black', linewidth=1)
        
        # 添加效率等级和数值标签
        for i, (bar, fps) in enumerate(zip(bars, fps_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{fps:.1f} FPS', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
            
            # 添加效率等级背景色
            if fps > 30:
                efficiency = "实时处理"
                bg_color = '#E8F5E8'
            elif fps > 15:
                efficiency = "准实时"
                bg_color = '#FFF3E0'
            else:
                efficiency = "离线处理"
                bg_color = '#FFEBEE'
            
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   efficiency, ha='center', va='center', 
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, alpha=0.8))
        
        ax.set_ylabel('推理速度 (帧/秒)', fontsize=12, fontweight='bold')
        ax.set_title('推理效率对比\n处理速度与实时性能分析', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(names, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加参考线
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='实时标准 (30 FPS)')
        ax.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='准实时标准 (15 FPS)')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'chart_efficiency.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_quality_vs_speed_chart(self, colors):
        """创建质量vs速度散点图表"""
        self.ensure_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, (name, data) in enumerate(self.data.items()):
            if data['psnr'] and data['total_time']:
                avg_psnr = np.mean(data['psnr'])
                avg_ssim = np.mean(data['ssim'])
                avg_time = np.mean(data['total_time'])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                # 散点大小根据SSIM调整
                size = avg_ssim * 300 + 100
                
                scatter = ax.scatter(fps, avg_psnr, s=size, label=name, 
                                   color=colors[i % len(colors)], alpha=0.7,
                                   edgecolors='black', linewidth=2)
                
                # 添加详细标注
                ax.annotate(f'{name}\nPSNR: {avg_psnr:.2f}dB\nSSIM: {avg_ssim:.4f}\n{fps:.1f} FPS', 
                           xy=(fps, avg_psnr), 
                           xytext=(15, 15), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor=colors[i % len(colors)], alpha=0.3,
                                   edgecolor=colors[i % len(colors)], linewidth=2),
                           arrowprops=dict(arrowstyle='->', 
                                         color=colors[i % len(colors)], lw=1.5))
        
        # 添加理想区域标注
        ax.axhspan(28, 35, alpha=0.1, color='green', label='高质量区域')
        ax.axvspan(15, 100, alpha=0.1, color='blue', label='实用速度区域')
        
        ax.set_xlabel('推理速度 (FPS)', fontsize=12, fontweight='bold')
        ax.set_ylabel('平均PSNR (dB)', fontsize=12, fontweight='bold')
        ax.set_title('质量vs速度权衡分析\n性能与效率的最佳平衡点', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='upper left')
        
        # 添加说明文本
        ax.text(0.98, 0.02, '气泡大小代表SSIM值\n右上角为理想区域', 
               transform=ax.transAxes, fontsize=10, 
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'chart_quality_vs_speed.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    # ... existing code ...
    def create_detailed_analysis(self):
        """创建详细分析图表"""
        if not self.data:
            print("没有数据可以创建详细分析")
            return
        
        # 确保中文字体设置
        self.ensure_chinese_font()
        
        # 创建详细分析图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # 1. 逐帧PSNR对比 (左上)
        ax1 = axes[0, 0]
        self.plot_frame_by_frame_comparison(ax1, 'psnr', 'PSNR (dB)', colors)
        
        # 2. 逐帧SSIM对比 (右上)
        ax2 = axes[0, 1]
        self.plot_frame_by_frame_comparison(ax2, 'ssim', 'SSIM', colors)
        
        # 3. 时间效率分析 (左下)
        ax3 = axes[1, 0]
        self.plot_efficiency_analysis(ax3, colors)
        
        # 4. 质量vs速度散点图 (右下)
        ax4 = axes[1, 1]
        self.plot_quality_vs_speed_old(ax4, colors)
        
        plt.tight_layout()
        plt.suptitle('Self-Blind-VSR 详细性能分析', fontsize=16, fontweight='bold', y=0.98)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'inference_analysis_detailed.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"详细分析图表已保存至: {output_path}")
        plt.show()
    
    def plot_frame_by_frame_comparison(self, ax, metric, ylabel, colors):
        """绘制逐帧对比图"""
        for i, (name, data) in enumerate(self.data.items()):
            if data[metric]:
                # 取前100帧进行可视化（避免图表过于密集）
                plot_data = data[metric][:100] if len(data[metric]) > 100 else data[metric]
                ax.plot(range(len(plot_data)), plot_data, label=name, 
                       color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('帧序号', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'逐帧{ylabel}对比 (前100帧)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_efficiency_analysis(self, ax, colors):
        """绘制效率分析图"""
        names = list(self.data.keys())
        fps_values = []
        
        for name in names:
            if self.data[name]['total_time']:
                avg_time = np.mean(self.data[name]['total_time'])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                fps_values.append(fps)
            else:
                fps_values.append(0)
        
        bars = ax.bar(names, fps_values, color=colors[:len(names)], alpha=0.8)
        ax.set_ylabel('推理速度 (FPS)', fontweight='bold')
        ax.set_title('推理效率对比 (帧/秒)', fontweight='bold')
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, fps in zip(bars, fps_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{fps:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    def plot_quality_vs_speed_old(self, ax, colors):
        """绘制质量vs速度散点图（原版本）"""
        for i, (name, data) in enumerate(self.data.items()):
            if data['psnr'] and data['total_time']:
                avg_psnr = np.mean(data['psnr'])
                avg_time = np.mean(data['total_time'])
                
                ax.scatter(avg_time, avg_psnr, s=100, label=name, 
                          color=colors[i % len(colors)], alpha=0.8)
                
                # 添加标签
                ax.annotate(name, (avg_time, avg_psnr), 
                           xytext=(5, 5), textcoords='offset points',
                           fontweight='bold')
        
        ax.set_xlabel('平均推理时间 (秒)', fontweight='bold')
        ax.set_ylabel('平均PSNR (dB)', fontweight='bold')
        ax.set_title('质量vs速度权衡分析', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*60)
        print("                推理性能统计摘要")
        print("="*60)
        
        for name, data in self.data.items():
            if data['psnr']:
                print(f"\n【{name}】")
                print(f"  总处理帧数: {len(data['psnr'])}")
                print(f"  平均PSNR: {np.mean(data['psnr']):.4f} dB")
                print(f"  PSNR标准差: {np.std(data['psnr']):.4f}")
                print(f"  平均SSIM: {np.mean(data['ssim']):.4f}")
                print(f"  SSIM标准差: {np.std(data['ssim']):.4f}")
                print(f"  平均推理时间: {np.mean(data['total_time']):.4f} 秒")
                print(f"  推理速度: {1.0/np.mean(data['total_time']):.2f} FPS")
                print(f"  前向推理时间占比: {np.mean(data['forward_time'])/np.mean(data['total_time'])*100:.1f}%")
                
                # 各视频统计
                print(f"  各视频详细统计:")
                for video, stats in data['video_stats'].items():
                    print(f"    视频{video}: PSNR={stats['avg_psnr']:.3f}, SSIM={stats['avg_ssim']:.4f}, 帧数={stats['total_frames']}")
        
        print("\n" + "="*60)

def main():
    # 清理matplotlib字体缓存，解决中文乱码问题
    try:
        import matplotlib.font_manager as fm
        fm._rebuild()
    except:
        pass
    
    # 强制设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 定义日志文件路径
    log_files = {
        'Jittor版本': 'jittor_results/infer_Realistic_REDS4/inference_log_2025-06-23 11-12-40.txt',
        'PyTorch版本': 'infer_results/infer_Realistic_REDS4/inference_log_2025-06-23 10-55-54.txt'
    }
    
    # 创建分析器并运行
    analyzer = LogAnalyzer(log_files)
    analyzer.load_all_data()
    
    # 生成多个独立图表（替代综合图表）
    analyzer.create_individual_charts()
    
    # 打印统计摘要
    analyzer.print_summary()

if __name__ == "__main__":
    main()