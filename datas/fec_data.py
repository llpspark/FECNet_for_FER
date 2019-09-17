'''FEC Dataset class'''

import pprint
import pandas as pd
import torch.utils.data as data
from PIL import Image

class FecData(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv("datas/pd_triplet_data.csv")
        self.data = self.pd_data.to_dict("list")
        self.data_anc = self.data['anchor']
        self.data_pos = self.data["postive"]
        self.data_neg = self.data["negative"]

    def __len__(self):
        return len(self.data["anchor"])

    def __getitem__(self, index):
        anc_img = Image.open(self.data_anc[index])
        pos_img = Image.open(self.data_pos[index])
        neg_img = Image.open(self.data_neg[index])
        if self.transform is not None:
            anc_img = self.transform(anc_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        return anc_img, pos_img, neg_img
