import torch
import imageio
import numpy as np
import os
import datetime
import skimage.color as sc
import shutil
import time

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Logger:
    def _safe_rename_directory(self, old_path, new_path, max_retries=3):
        """
        安全地重命名目录，处理Windows权限问题
        """
        # 如果目标路径已存在，直接跳过
        if os.path.exists(new_path):
            print(f"目标路径已存在，跳过重命名: {new_path}")
            return
            
        # 尝试直接重命名
        for attempt in range(max_retries):
            try:
                os.rename(old_path, new_path)
                print(f"✅ 成功重命名: {old_path} -> {new_path}")
                return
                
            except PermissionError as e:
                print(f"⚠️  重命名尝试 {attempt + 1}/{max_retries} 失败: 权限被拒绝")
                if attempt < max_retries - 1:
                    time.sleep(2)  # 等待2秒再重试
                    
            except Exception as e:
                print(f"⚠️  重命名尝试 {attempt + 1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        # 所有直接重命名尝试都失败，尝试复制+删除方式
        try:
            print("🔄 尝试使用复制+删除的方式...")
            
            # 确保目标目录不存在
            if os.path.exists(new_path):
                shutil.rmtree(new_path)
                
            # 复制整个目录树
            shutil.copytree(old_path, new_path)
            print(f"✅ 复制完成: {old_path} -> {new_path}")
            
            # 删除原目录
            shutil.rmtree(old_path)
            print(f"✅ 成功通过复制方式重命名完成")
            return
            
        except Exception as copy_error:
            print(f"❌ 复制方式也失败: {copy_error}")
            print(f"💡 将继续使用原目录: {old_path}")
            print("   这不会影响训练过程，只是不会归档旧的实验结果")
            return

    def __init__(self, args):
        self.args = args
        self.psnr_log = torch.Tensor()
        self.loss_log = torch.Tensor()

        if args.load == '.':
            if args.save == '.':
                args.save = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
            else:
                # 为指定的save名称添加时间戳
                args.save = args.save + '_' + datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
            # 如果目录已存在，直接生成新的唯一目录名
            original_dir = args.experiment_dir + args.save
            if os.path.exists(original_dir) and not args.test_only:
                # 生成唯一的新目录名
                # 使用秒级时间戳确保唯一性
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
                self.dir = f"{original_dir}_{timestamp}"
                if os.path.exists(self.dir):
                    counter = 1
                    while True:
                        self.dir = f"{original_dir}_{timestamp}_{counter:02d}"
                        if not os.path.exists(self.dir):
                            break
                        counter += 1
                print(f"🔄 目录已存在，创建新目录: {self.dir}")
            else:
                self.dir = original_dir
        else:
            self.dir = args.experiment_dir + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.loss_log = torch.load(self.dir + '/loss_log.pt')[:, -1]
                self.psnr_log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.psnr_log)))

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        if not os.path.exists(self.dir + '/model'):
            os.makedirs(self.dir + '/model')

        if not os.path.exists(self.dir + '/result/' + self.args.data_test):
            print("Creating dir for saving images...", self.dir + '/result/' + self.args.data_test)
            os.makedirs(self.dir + '/result/' + self.args.data_test)

        print('Save Path : {}'.format(self.dir))

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write('From epoch {}...'.format(len(self.psnr_log)) + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')

    def save(self, trainer, epoch, is_best):
        trainer.model.save(self.dir, epoch, is_best)
        torch.save(self.psnr_log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        self.plot_psnr_log(epoch)

    def save_images(self, filename, save_list, epoch):
        if self.args.task == 'FlowVideoSR':
            f = filename.split('.')
            dirname = '{}/result/{}/{}'.format(self.dir, self.args.data_test, f[0])
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            filename = '{}/{}'.format(dirname, f[1])
            postfix = ['gt', 'lr', 'sr', 'lr_cycle', 'lr_down', 'sr_down', 'est_kernel', 'gt_kernel']
        else:
            raise NotImplementedError('Task [{:s}] is not found'.format(self.args.task))
        for img, post in zip(save_list, postfix):
            img = img[0].data
            img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            if img.shape[2] == 1:
                img = img.squeeze(axis=2)
            elif img.shape[2] == 3 and self.args.n_colors == 1:
                img = sc.ycbcr2rgb(img.astype('float')).clip(0, 1)
                img = (255 * img).round().astype('uint8')
            imageio.imwrite('{}_{}.png'.format(filename, post), img)

    def start_log(self, train=True):
        if train:
            self.loss_log = torch.cat((self.loss_log, torch.zeros(1)))
        else:
            self.psnr_log = torch.cat((self.psnr_log, torch.zeros(1)))

    def report_log(self, item, train=True):
        if train:
            self.loss_log[-1] += item
        else:
            self.psnr_log[-1] += item

    def end_log(self, n_div, train=True):
        if train:
            self.loss_log[-1].div_(n_div)
        else:
            self.psnr_log[-1].div_(n_div)

    def plot_loss_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('Loss Graph')
        plt.plot(axis, self.loss_log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'loss.pdf'))
        plt.close(fig)

    def plot_psnr_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('PSNR Graph')
        plt.plot(axis, self.psnr_log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'psnr.pdf'))
        plt.close(fig)

    def plot_log(self, data_list, filename, title):
        epoch = len(data_list)
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('{} Graph'.format(title))
        plt.plot(axis, np.array(data_list))
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, filename))
        plt.close(fig)

    def done(self):
        self.log_file.close()
