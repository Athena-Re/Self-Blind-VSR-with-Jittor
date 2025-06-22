import os
from importlib import import_module

import jittor as jt
import jittor.nn as nn


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.cpu = args.cpu
        self.n_GPUs = args.n_GPUs
        self.save_middle_models = args.save_middle_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args)

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.get_model(), file=ckp.log_file)

    def execute(self, *args):
        return self.model(*args)

    def get_model(self):
        return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        jt.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pkl')
        )
        if is_best:
            jt.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pkl')
            )
        if self.save_middle_models:
            if epoch % 1 == 0:
                jt.save(
                    target.state_dict(),
                    os.path.join(apath, 'model', 'model_{}.pkl'.format(epoch))
                )

    def load(self, apath, pre_train='.', resume=False, cpu=False):
        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(jt.load(pre_train))
        elif resume:
            print('Loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pkl')))
            self.get_model().load_state_dict(jt.load(os.path.join(apath, 'model', 'model_latest.pkl')))
        elif self.args.test_only:
            self.get_model().load_state_dict(jt.load(os.path.join(apath, 'model', 'model_best.pkl')))
        else:
            pass 