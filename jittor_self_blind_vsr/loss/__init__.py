import os
from importlib import import_module
import numpy as np
import jittor as jt
import jittor.nn as nn
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Loss(nn.Module):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')()
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found'.format(loss_type))

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = jt.Var([])

        if args.load != '.':
            self.load(ckp.dir, cpu=args.cpu)

    def execute(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                if len(self.log) > 0:
                    self.log[-1, i] += effective_loss.data[0]
            elif l['type'] == 'DIS':
                if len(self.log) > 0:
                    self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1 and len(self.log) > 0:
            self.log[-1, -1] += loss_sum.data[0]

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = jt.concat((self.log, jt.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        if len(self.log) > 0:
            self.log[-1] = self.log[-1] / n_batches

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def get_loss_module(self):
        return self.loss_module

    def save(self, apath):
        jt.save(self.state_dict(), os.path.join(apath, 'loss.pkl'))
        jt.save(self.log, os.path.join(apath, 'loss_log.pkl'))

    def load(self, apath, cpu=False):
        self.load_state_dict(jt.load(os.path.join(apath, 'loss.pkl')))
        self.log = jt.load(os.path.join(apath, 'loss_log.pkl'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): 
                    l.scheduler.step() 