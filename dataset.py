import os
import pickle
from suppor_lib import image_process


class JUFE_Dataset():
    def __init__(
        self,
        path,
        img_list,
        mos_list,
    ):
        self.path = path
        self.img_list = img_list
        self.mos_list = mos_list

        # access the starting points of viewing
        scanpaths = pickle.load(open(os.path.join(path, 'scanpaths.pkl'), 'rb'))
        self.scanpaths = scanpaths

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        # get key value for searching label, starting_point, and exploration_time.
        key = self.img_list[index]

        # ground-truth label
        mos = self.mos_list[key]

        # handle with key value
        # an example of key value: '2_len2_bd_1_good_5s' (with good/bad indicates different starting points)
        _split = key.split('_')
        sp_key = _split[0] + '_' + _split[1] + '_' + \
            _split[2] + '_' + _split[3] + '_' + _split[4]
        name_with_png = _split[0] + '_' + _split[1] + \
            '_' + _split[2] + '_' + _split[3] + '.png'

        # get the starting points for prediction
        starting_point = self.scanpaths[sp_key][0, 0, :]

        # get image tensors of raw and downsampled images
        raw_img, ds_img = image_process(os.path.join(self.path, 'imgs', name_with_png))

        exploration_time = int(_split[-1].split('s')[0])

        return raw_img, ds_img, mos, starting_point, exploration_time


class CVIQ_Dataset():
    def __init__(
        self,
        path,
        img_list,
        mos_list,
    ):
        self.path = path
        self.img_list = img_list
        self.mos_list = mos_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        mos = self.mos_list[self.img_list[index]]

        name = self.img_list[index] + '.png'

        raw_img, ds_img = image_process(os.path.join(self.path, 'imgs', name))

        return raw_img, ds_img, mos, 0, 20


class OIQA_Dataset():
    def __init__(
        self,
        path,
        img_list,
        mos_list,
    ):
        self.path = path
        self.img_list = img_list
        self.mos_list = mos_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        mos = self.mos_list[self.img_list[index]]

        name = 'img' + self.img_list[index] + '.png'
        if not os.path.exists('./Dataset/OIQA/imgs/' + name):
            name = 'img' + self.img_list[index] + '.jpg'

        raw_img, ds_img = image_process(
            os.path.join(self.path, 'imgs', name), resize=True)

        return raw_img, ds_img, mos, 0, 20


