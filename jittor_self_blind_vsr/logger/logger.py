import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import jittor as jt


class Logger:
    def __init__(self, args):
        self.args = args
        self.n_processes = 8

        if args.load == '.':
            if args.save == '.': args.save = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

            self.dir = os.path.join(args.experiment_dir, args.save)
        else:
            self.dir = os.path.join(args.experiment_dir, args.load)
            if os.path.exists(self.dir):
                self.log = jt.load(self.get_path('psnr_log.pkl'))
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.get_path('model'))
        _make_dir(self.get_path('results'))

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now() + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        jt.save(self.log, self.get_path('psnr_log.pkl'))

    def add_log(self, log):
        self.log = jt.concat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def plot_log(self, log, filename='loss.pdf', title='Loss'):
        fig = plt.figure()
        plt.title(title)
        plt.plot(log)
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.grid(True)
        plt.savefig(self.get_path(filename))
        plt.close(fig)

    def start_log(self, train=True):
        if train:
            self.loss_log = jt.zeros(1)
        else:
            self.psnr_log = jt.zeros(1)

    def report_log(self, item, train=True):
        if train:
            self.loss_log[-1] += item
        else:
            self.psnr_log[-1] += item

    def end_log(self, n_batches, train=True):
        if train:
            self.loss_log[-1].div_(n_batches)
        else:
            self.psnr_log[-1].div_(n_batches)
            self.psnr_log = jt.concat([self.psnr_log, jt.zeros(1)])

    def save_images(self, filename, save_list, epoch):
        if self.args.save_images:
            filename = self.get_path('results', '{}_x{}_'.format(filename, self.args.scale[0]))
            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].data.mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                imageio.imwrite('{}{}.png'.format(filename, p), tensor_cpu.numpy())


def now():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') 