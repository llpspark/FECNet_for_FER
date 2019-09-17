import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.Convolution1 = self.__conv(2, name='Convolution1', in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm1 = self.__batch_normalization(2, 'BatchNorm1', num_features=16, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution2 = self.__conv(2, name='Convolution2', in_channels=16, out_channels=12, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm2 = self.__batch_normalization(2, 'BatchNorm2', num_features=28, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution3 = self.__conv(2, name='Convolution3', in_channels=28, out_channels=12, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm3 = self.__batch_normalization(2, 'BatchNorm3', num_features=40, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution4 = self.__conv(2, name='Convolution4', in_channels=40, out_channels=12, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm4 = self.__batch_normalization(2, 'BatchNorm4', num_features=52, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution5 = self.__conv(2, name='Convolution5', in_channels=52, out_channels=12, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm5 = self.__batch_normalization(2, 'BatchNorm5', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution6 = self.__conv(2, name='Convolution6', in_channels=64, out_channels=12, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm6 = self.__batch_normalization(2, 'BatchNorm6', num_features=76, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution7 = self.__conv(2, name='Convolution7', in_channels=76, out_channels=12, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm13 = self.__batch_normalization(2, 'BatchNorm13', num_features=88, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution14 = self.__conv(2, name='Convolution14', in_channels=88, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm14 = self.__batch_normalization(2, 'BatchNorm14', num_features=96, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution15 = self.__conv(2, name='Convolution15', in_channels=96, out_channels=12, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm15 = self.__batch_normalization(2, 'BatchNorm15', num_features=108, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution16 = self.__conv(2, name='Convolution16', in_channels=108, out_channels=12, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm16 = self.__batch_normalization(2, 'BatchNorm16', num_features=120, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution17 = self.__conv(2, name='Convolution17', in_channels=120, out_channels=12, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm17 = self.__batch_normalization(2, 'BatchNorm17', num_features=132, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution18 = self.__conv(2, name='Convolution18', in_channels=132, out_channels=12, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.BatchNorm26 = self.__batch_normalization(2, 'BatchNorm26', num_features=144, eps=9.999999747378752e-06, momentum=0.0)
        self.Convolution27 = self.__conv(2, name='Convolution27', in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=False)
        self.fc6_1 = self.__dense(name = 'fc6_1', in_features = 96, out_features = 29, bias = True)
        self.fc8_1 = self.__dense(name = 'fc8_1', in_features = 29, out_features = 4, bias = True)

    def forward(self, x):
        Convolution1_pad = F.pad(x, (1, 1, 1, 1))
        Convolution1    = self.Convolution1(Convolution1_pad)
        BatchNorm1      = self.BatchNorm1(Convolution1)
        ReLU1           = F.relu(BatchNorm1)
        Convolution2_pad = F.pad(ReLU1, (1, 1, 1, 1))
        Convolution2    = self.Convolution2(Convolution2_pad)
        Dropout1        = F.dropout(input = Convolution2, p = 0.20000000298023224, training = self.training, inplace = True)
        Concat1         = torch.cat((Convolution1, Dropout1), 1)
        BatchNorm2      = self.BatchNorm2(Concat1)
        ReLU2           = F.relu(BatchNorm2)
        Convolution3_pad = F.pad(ReLU2, (1, 1, 1, 1))
        Convolution3    = self.Convolution3(Convolution3_pad)
        Dropout2        = F.dropout(input = Convolution3, p = 0.20000000298023224, training = self.training, inplace = True)
        Concat2         = torch.cat((Concat1, Dropout2), 1)
        BatchNorm3      = self.BatchNorm3(Concat2)
        ReLU3           = F.relu(BatchNorm3)
        Convolution4_pad = F.pad(ReLU3, (1, 1, 1, 1))
        Convolution4    = self.Convolution4(Convolution4_pad)
        Dropout3        = F.dropout(input = Convolution4, p = 0.20000000298023224, training = self.training, inplace = True)
        Concat3         = torch.cat((Concat2, Dropout3), 1)
        BatchNorm4      = self.BatchNorm4(Concat3)
        ReLU4           = F.relu(BatchNorm4)
        Convolution5_pad = F.pad(ReLU4, (1, 1, 1, 1))
        Convolution5    = self.Convolution5(Convolution5_pad)
        Dropout4        = F.dropout(input = Convolution5, p = 0.20000000298023224, training = self.training, inplace = True)
        Concat4         = torch.cat((Concat3, Dropout4), 1)
        BatchNorm5      = self.BatchNorm5(Concat4)
        ReLU5           = F.relu(BatchNorm5)
        Convolution6_pad = F.pad(ReLU5, (1, 1, 1, 1))
        Convolution6    = self.Convolution6(Convolution6_pad)
        Dropout5        = F.dropout(input = Convolution6, p = 0.20000000298023224, training = self.training, inplace = True)
        Concat5         = torch.cat((Concat4, Dropout5), 1)
        BatchNorm6      = self.BatchNorm6(Concat5)
        ReLU6           = F.relu(BatchNorm6)
        Convolution7_pad = F.pad(ReLU6, (1, 1, 1, 1))
        Convolution7    = self.Convolution7(Convolution7_pad)
        Dropout6        = F.dropout(input = Convolution7, p = 0.20000000298023224, training = self.training, inplace = True)
        Concat6         = torch.cat((Concat5, Dropout6), 1)
        BatchNorm13     = self.BatchNorm13(Concat6)
        ReLU13          = F.relu(BatchNorm13)
        Convolution14   = self.Convolution14(ReLU13)
        Dropout13       = F.dropout(input = Convolution14, p = 0.20000000298023224, training = self.training, inplace = True)
        Pooling1        = F.avg_pool2d(Dropout13, kernel_size=(2, 2), stride=(2, 2), padding=(0,), ceil_mode=True, count_include_pad=False)
        BatchNorm14     = self.BatchNorm14(Pooling1)
        ReLU14          = F.relu(BatchNorm14)
        Convolution15_pad = F.pad(ReLU14, (1, 1, 1, 1))
        Convolution15   = self.Convolution15(Convolution15_pad)
        Dropout14       = F.dropout(input = Convolution15, p = 0.20000000298023224, training = self.training, inplace = True)
        Concat13        = torch.cat((Pooling1, Dropout14), 1)
        BatchNorm15     = self.BatchNorm15(Concat13)
        ReLU15          = F.relu(BatchNorm15)
        Convolution16_pad = F.pad(ReLU15, (1, 1, 1, 1))
        Convolution16   = self.Convolution16(Convolution16_pad)
        Dropout15       = F.dropout(input = Convolution16, p = 0.20000000298023224, training = self.training, inplace = True)
        Concat14        = torch.cat((Concat13, Dropout15), 1)
        BatchNorm16     = self.BatchNorm16(Concat14)
        ReLU16          = F.relu(BatchNorm16)
        Convolution17_pad = F.pad(ReLU16, (1, 1, 1, 1))
        Convolution17   = self.Convolution17(Convolution17_pad)
        Dropout16       = F.dropout(input = Convolution17, p = 0.20000000298023224, training = self.training, inplace = True)
        Concat15        = torch.cat((Concat14, Dropout16), 1)
        BatchNorm17     = self.BatchNorm17(Concat15)
        ReLU17          = F.relu(BatchNorm17)
        Convolution18_pad = F.pad(ReLU17, (1, 1, 1, 1))
        Convolution18   = self.Convolution18(Convolution18_pad)
        Dropout17       = F.dropout(input = Convolution18, p = 0.20000000298023224, training = self.training, inplace = True)
        Concat16        = torch.cat((Concat15, Dropout17), 1)
        BatchNorm26     = self.BatchNorm26(Concat16)
        ReLU26          = F.relu(BatchNorm26)
        Convolution27_pad = F.pad(ReLU26, (1, 1, 1, 1))
        Convolution27   = self.Convolution27(Convolution27_pad)
        Dropout26       = F.dropout(input = Convolution27, p = 0.20000000298023224, training = self.training, inplace = True)
        pool3           = F.max_pool2d(Dropout26, kernel_size=(8, 8), stride=(1, 1), padding=0, ceil_mode=False)
        fc6_0           = pool3.view(pool3.size(0), -1)
        fc6_1           = self.fc6_1(fc6_0)
        relu6           = F.relu(fc6_1)
        drop6           = F.dropout(input = relu6, p = 0.5, training = self.training, inplace = True)
        fc8_0           = drop6.view(drop6.size(0), -1)
        fc8_1           = self.fc8_1(fc8_0)
        out             = F.softmax(fc8_1)
        return out


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

