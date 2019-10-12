'''FEC Dataset class'''

import pprint
import pandas as pd
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class FecData(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv("datas/pd_triplet_data_new.csv")
        self.data = self.pd_data.to_dict("list")
        self.data_anc = self.data['anchor']
        self.data_pos = self.data["postive"]
        self.data_neg = self.data["negative"]

    def __len__(self):
        return len(self.data["anchor"])

    def __getitem__(self, index):
        anc_list = eval(self.data_anc[index])
        anc_img = Image.open(anc_list[0])
        wid, hei = anc_img.size
        anc_img = anc_img.crop((anc_list[1][0] * wid, anc_list[1][2] * hei, anc_list[1][1] * wid, anc_list[1][3] * hei))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')

        pos_list = eval(self.data_pos[index])
        pos_img = Image.open(pos_list[0])
        wid, hei = pos_img.size
        pos_img = pos_img.crop((pos_list[1][0] * wid, pos_list[1][2] * hei, pos_list[1][1] * wid, pos_list[1][3] * hei))
        if pos_img.getbands()[0] != 'R':
            pos_img = pos_img.convert('RGB')

        neg_list = eval(self.data_neg[index])
        neg_img = Image.open(neg_list[0])
        wid, hei = neg_img.size
        neg_img = neg_img.crop((neg_list[1][0] * wid, neg_list[1][2] * hei, neg_list[1][1] * wid, neg_list[1][3] * hei))
        if neg_img.getbands()[0] != 'R':
            neg_img = neg_img.convert('RGB')


        if self.transform is not None:
            anc_img = self.transform(anc_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        return anc_img, pos_img, neg_img


class FecTestData(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, transform=None):
        self.transform = transform
        self.pd_test_data = pd.read_csv("datas/pd_triplet_data_test.csv")
        self.data = self.pd_test_data.to_dict("list")
        self.data_anc = self.data['anchor']
        self.data_pos = self.data["postive"]
        self.data_neg = self.data["negative"]

    def __len__(self):
        return len(self.data["anchor"])

    def __getitem__(self, index):
        anc_list = eval(self.data_anc[index])
        anc_img = Image.open(anc_list[0])
        wid, hei = anc_img.size
        anc_img = anc_img.crop((anc_list[1][0] * wid, anc_list[1][2] * hei, anc_list[1][1] * wid, anc_list[1][3] * hei))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')

        pos_list = eval(self.data_pos[index])
        pos_img = Image.open(pos_list[0])
        wid, hei = pos_img.size
        pos_img = pos_img.crop((pos_list[1][0] * wid, pos_list[1][2] * hei, pos_list[1][1] * wid, pos_list[1][3] * hei))
        if pos_img.getbands()[0] != 'R':
            pos_img = pos_img.convert('RGB')

        neg_list = eval(self.data_neg[index])
        neg_img = Image.open(neg_list[0])
        wid, hei = neg_img.size
        neg_img = neg_img.crop((neg_list[1][0] * wid, neg_list[1][2] * hei, neg_list[1][1] * wid, neg_list[1][3] * hei))
        if neg_img.getbands()[0] != 'R':
            neg_img = neg_img.convert('RGB')


        if self.transform is not None:
            anc_img = self.transform(anc_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        return anc_img, pos_img, neg_img
