import os
from data.videodata_online_gaussian import VIDEODATA_ONLINE_GAUSSIAN


class REDS_HRLR(VIDEODATA_ONLINE_GAUSSIAN):
    def __init__(self, args, name='REDS_HRLR', train=False):
        super(REDS_HRLR, self).__init__(args, name=name, train=train)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'gt')
        print("DataSet gt path:", self.dir_gt) 