import torch
from torch.utils import data
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import datas
from models.dense2 import KitModel
import torchvision.transforms.transforms as transforms
import importlib

use_cuda = torch.cuda.is_available()
total_epoch = 80

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize([32, 32]),
    transforms.ToTensor()])

trainset = datas.fec_data.FecData(transform)
trainloader = data.DataLoader(trainset, batch_size=256, num_workers=16)

net = KitModel()
if use_cuda:
    net.cuda()

init_lr = 0.0001
criterion = nn.TripletMarginLoss(margin=0.2)
optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)

# Training
total_epoch = 100

def train(epoch):
    print("train {} epoch".format(epoch))
    net.train()
    for batch_idx, (anc, pos, neg) in enumerate(trainloader):
        if use_cuda:
            anc, pos, neg = anc.cuda(), pos.cuda(), neg.cuda()
        optimizer.zero_grad()
        anc, pos, neg = Variable(anc), Variable(pos), Variable(neg)
        anc_fea = net(anc)
        pos_fea = net(pos)
        neg_fea = net(neg)
        loss = criterion(anc_fea, pos_fea, neg_fea)
        print(loss)
        loss.backward()
        optimizer.step()
    if epoch % 2 == 0:
        torch.save(net, "./check_points/{}_checkpoint.pkl".format(epoch))
        print("save {}_checkpoint.pkl".format(epoch))

for i in range(total_epoch):
    train(i)



