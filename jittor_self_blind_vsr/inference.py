# python inference.py --input_path ./test_data --model_path ./pretrained_model.pkl
import os
import sys
import argparse
import time
import math
import glob
import imageio
import cv2
import numpy as np

import warnings

warnings.filterwarnings("ignore")

# 添加模型路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jittor as jt
from model.pwc_recons import PWC_Recons


# 设置Jittor使用GPU
jt.flags.use_cuda = 1 if jt.has_cuda else 0

class Logger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file = os.path.join(result_dir, filename)
        self.logger = open(self.log_file, 'w')

    def write_log(self, log):
        print(log)  # 同时输出到控制台
        self.logger.write(log + '\n')
        self.logger.flush()


class Inference:
    def __init__(self, args):

        self.input_path = args.input_path
        self.GT_path = args.gt_path
        self.model_path = args.model_path
        self.result_path = args.result_path
        self.dataset_name = args.dataset_name
        
        # 根据模型文件名自动推断blur_type
        model_filename = os.path.basename(self.model_path).lower()
        if 'realistic' in model_filename:
            self.blur_type = 'Realistic'
        elif 'gaussian' in model_filename:
            self.blur_type = 'Gaussian'
        else:
            # 如果无法从文件名推断，则使用命令行参数
            self.blur_type = args.blur_type
            
        self.border = args.border
        self.save_image = args.save_image
        self.n_seq = args.n_seq
        self.scale = args.scale
        self.size_must_mode = args.size_must_mode

        # 创建基础结果目录
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path, exist_ok=True)
            print('mkdir: {}'.format(self.result_path))

        # 根据数据集名称和模糊类型创建子目录
        dataset_folder = f'infer_{self.blur_type}_{self.dataset_name}'
        self.result_path = os.path.join(self.result_path, dataset_folder)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path, exist_ok=True)
            print('mkdir: {}'.format(self.result_path))

        time_str = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        self.logger = Logger(self.result_path, 'inference_log_{}.txt'.format(time_str))

        # 显示GPU状态
        if jt.has_cuda and jt.flags.use_cuda:
            self.logger.write_log('使用GPU进行推理')
            self.logger.write_log(f'GPU设备数量: {jt.get_device_count()}')
        else:
            self.logger.write_log('使用CPU进行推理')

        # 显示模糊类型推断信息
        model_filename = os.path.basename(self.model_path).lower()
        if 'realistic' in model_filename or 'gaussian' in model_filename:
            self.logger.write_log(f'从模型文件名自动推断blur_type: {self.blur_type}')
        else:
            self.logger.write_log(f'使用命令行参数blur_type: {self.blur_type}')

        self.logger.write_log('Inference - {} {}'.format(self.blur_type, self.dataset_name))
        self.logger.write_log('dataset_name: {}'.format(self.dataset_name))
        self.logger.write_log('blur_type: {}'.format(self.blur_type))
        self.logger.write_log('input_path: {}'.format(self.input_path))
        self.logger.write_log('gt_path: {}'.format(self.GT_path))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        self.logger.write_log('border: {}'.format(self.border))
        self.logger.write_log('save_image: {}'.format(self.save_image))
        self.logger.write_log('n_seq: {}'.format(self.n_seq))
        self.logger.write_log('scale: {}'.format(self.scale))
        self.logger.write_log('size_must_mode: {}'.format(self.size_must_mode))

        # 加载模型
        self.logger.write_log('正在加载模型...')
        self.net = PWC_Recons(
            n_colors=3, n_sequence=5, extra_RBS=3, recons_RBS=20, n_feat=128, scale=4, ksize=13
        )
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'模型文件不存在: {self.model_path}')
            
        self.net.load_state_dict(jt.load(self.model_path))
        self.logger.write_log('模型加载完成: {}'.format(self.model_path))
        self.net.eval()

        # 预热模型
        self.logger.write_log('正在预热模型...')
        dummy_input = jt.randn(1, 5, 3, 64, 64)
        with jt.no_grad():
            _ = self.net({'x': dummy_input, 'mode': 'infer'})
        self.logger.write_log('模型预热完成')

    def infer(self):
        with jt.no_grad():
            total_psnr = {}
            total_ssim = {}
            videos = sorted(os.listdir(self.input_path))
            
            self.logger.write_log(f'找到 {len(videos)} 个视频序列: {videos}')
            
            # 计算总的序列数量
            total_sequences = 0
            for v in videos:
                input_frames = sorted(glob.glob(os.path.join(self.input_path, v, "*.png")))
                input_seqs = self.gene_seq(input_frames, n_seq=self.n_seq)
                total_sequences += len(input_seqs)
                self.logger.write_log(f'视频 {v}: {len(input_frames)} 帧, {len(input_seqs)} 个序列')
            
            self.logger.write_log(f'总共需要处理 {total_sequences} 个序列')
            
            for v in videos:
                video_psnr = []
                video_ssim = []
                input_frames = sorted(glob.glob(os.path.join(self.input_path, v, "*.png")))
                gt_frames = sorted(glob.glob(os.path.join(self.GT_path, v, "*.png")))
                input_seqs = self.gene_seq(input_frames, n_seq=self.n_seq)
                gt_seqs = self.gene_seq(gt_frames, n_seq=self.n_seq)
                
                self.logger.write_log(f"开始处理视频: {v}")
                
                for seq_idx, (in_seq, gt_seq) in enumerate(zip(input_seqs, gt_seqs)):
                    start_time = time.time()
                    filename = os.path.basename(in_seq[self.n_seq // 2]).split('.')[0]
                    
                    # 读取图像
                    inputs = [imageio.imread(p) for p in in_seq]
                    gt = imageio.imread(gt_seq[self.n_seq // 2])

                    h, w, c = inputs[self.n_seq // 2].shape
                    new_h, new_w = h - h % self.size_must_mode, w - w % self.size_must_mode
                    inputs = [im[:new_h, :new_w, :] for im in inputs]
                    gt = gt[:new_h * self.scale, :new_w * self.scale, :]

                    # 转换为tensor
                    in_tensor = self.numpy2tensor(inputs)
                    preprocess_time = time.time()
                    
                    # 前向传播
                    try:
                        output_dict, _ = self.net({'x': in_tensor, 'mode': 'infer'})
                        output = output_dict['recons']
                    except Exception as e:
                        self.logger.write_log(f'前向传播错误: {e}')
                        continue
                        
                    forward_time = time.time()
                    output_img = self.tensor2numpy(output)

                    psnr, ssim = self.get_PSNR_SSIM(output_img, gt)
                    video_psnr.append(psnr)
                    video_ssim.append(ssim)
                    total_psnr[v] = video_psnr
                    total_ssim[v] = video_ssim

                    if self.save_image:
                        if not os.path.exists(os.path.join(self.result_path, v)):
                            os.mkdir(os.path.join(self.result_path, v))
                        imageio.imwrite(os.path.join(self.result_path, v, '{}.png'.format(filename)), output_img)
                    postprocess_time = time.time()

                    self.logger.write_log(
                        '> {}-{} PSNR={:.5}, SSIM={:.4} pre_time:{:.3}s, forward_time:{:.3}s, post_time:{:.3}s, total_time:{:.3}s'
                            .format(v, filename, psnr, ssim,
                                    preprocess_time - start_time,
                                    forward_time - preprocess_time,
                                    postprocess_time - forward_time,
                                    postprocess_time - start_time))
                    



            
            sum_psnr = 0.
            sum_ssim = 0.
            n_img = 0
            for k in total_psnr.keys():
                self.logger.write_log("# Video:{} AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(
                    k, sum(total_psnr[k]) / len(total_psnr[k]), sum(total_ssim[k]) / len(total_ssim[k])))
                sum_psnr += sum(total_psnr[k])
                sum_ssim += sum(total_ssim[k])
                n_img += len(total_psnr[k])
            self.logger.write_log("# Total AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(sum_psnr / n_img, sum_ssim / n_img))

    def gene_seq(self, img_list, n_seq):
        if self.border:
            half = n_seq // 2
            img_list_temp = img_list[:half]
            img_list_temp.extend(img_list)
            img_list_temp.extend(img_list[-half:])
            img_list = img_list_temp
        seq_list = []
        for i in range(len(img_list) - 2 * (n_seq // 2)):
            seq_list.append(img_list[i:i + n_seq])
        return seq_list

    def numpy2tensor(self, input_seq, rgb_range=1.):
        tensor_list = []
        for img in input_seq:
            img = np.array(img).astype('float64')
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
            tensor = jt.array(np_transpose).float()  # numpy -> tensor
            tensor = tensor * (rgb_range / 255)  # (0,255) -> (0,1)
            tensor_list.append(tensor)
        stacked = jt.stack(tensor_list).unsqueeze(0)
        return stacked

    def tensor2numpy(self, tensor, rgb_range=1.):
        rgb_coefficient = 255 / rgb_range
        img = tensor * rgb_coefficient
        img = jt.clamp(img, 0, 255).round()
        img = img[0].data
        img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        return img

    def get_PSNR_SSIM(self, output, gt, crop_border=4):
        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_GT = gt[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = self.calc_PSNR(cropped_GT, cropped_output)
        ssim = self.calc_SSIM(cropped_GT, cropped_output)
        return psnr, ssim

    def calc_PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calc_SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''

        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jittor Self-Blind-VSR Inference')

    # Basic settings
    parser.add_argument('--input_path', type=str, default='../dataset/input',
                        help='Path to input LR videos')
    parser.add_argument('--gt_path', type=str, default='../dataset/gt',
                        help='Path to ground truth HR videos')
    parser.add_argument('--model_path', type=str, default='../pretrain_models/self_blind_vsr_realistic_numpy.pkl',
                        help='Path to pretrained model')
    parser.add_argument('--result_path', type=str, default='../jittor_results',
                        help='Path to save results')
    parser.add_argument('--dataset_name', type=str, default='REDS4',
                        help='Dataset name for result folder naming (e.g., REDS4, Vid4, SPMCS)')
    parser.add_argument('--blur_type', type=str, default='Gaussian', choices=['Gaussian', 'Realistic'],
                        help='Blur type: Gaussian or Realistic (自动从模型文件名推断，如包含realistic/gaussian关键词)')
    parser.add_argument('--infer_flag', type=str, default='.',
                        help='Inference flag for result folder naming')

    # Model settings
    parser.add_argument('--n_seq', type=int, default=5,
                        help='Number of input sequence frames')
    parser.add_argument('--scale', type=int, default=4,
                        help='Super resolution scale')
    parser.add_argument('--size_must_mode', type=int, default=4,
                        help='Size must be divisible by this number')

    # Processing settings
    parser.add_argument('--border', action='store_true',
                        help='Add border frames for sequence generation')
    parser.add_argument('--save_image', action='store_true', default=True,
                        help='Save output images')

    args = parser.parse_args()

    # Set Jittor flags for better performance
    jt.flags.use_cuda = 1 if jt.has_cuda else 0

    inference = Inference(args)
    inference.infer() 