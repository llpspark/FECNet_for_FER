import imp
import torch
import numpy as np
from torch.utils import data
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import datas
from models.v2_net import KitModel
import torchvision.transforms.transforms as transforms
import importlib
from torch.nn.modules.distance import PairwiseDistance
from eval_metrics import evaluate, plot_roc

l2_dist = PairwiseDistance(2)
use_cuda = torch.cuda.is_available()
total_epoch = 150

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize([32, 32]),
    transforms.ToTensor()])

trainset = datas.fec_data.FecData(transform)
trainloader = data.DataLoader(trainset, batch_size=256, num_workers=16)

# net = KitModel()

# load model which converted from caffe model using MMdnn tool
MainModel = imp.load_source('MainModel', "./models/v2_net.py")
net = torch.load("./models/v2_net.pth")
net.fc8_1 = nn.Linear(net.fc8_1.in_features, 4)

for param in net.parameters():
    param.requiers_grad = False

for param in net.fc8_1.parameters():
    param.requiers_grad = True


if use_cuda:
    net.cuda()

init_lr = 0.0005
criterion = nn.TripletMarginLoss(margin=0.2)
optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)

# Training
total_epoch = 100

def train(epoch):
    print("train {} epoch".format(epoch))
    labels, distances = [], []
    triplet_loss_sum = 0.0
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
        print(loss.item())
        loss.backward()
        optimizer.step()

        dists = l2_dist.forward(anc_fea, pos_fea)
        distances.append(dists.data.cpu().numpy())
        labels.append(np.ones(dists.size(0)))

        dists = l2_dist.forward(anc_fea, neg_fea)
        distances.append(dists.data.cpu().numpy())
        labels.append(np.zeros(dists.size(0)))
        triplet_loss_sum += loss.item()

    avg_triplet_loss = triplet_loss_sum / trainset.__len__()
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    print(labels)
    print(distances)
    tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
    print('  train set - finetune Triplet Loss       = {:.8f}'.format(avg_triplet_loss))
    print('  train set - finetune Accuracy           = {:.8f}'.format(np.mean(accuracy)))
    with open("./log/tune_v2_4.log", "a+") as log_file:
        log_file.write("epoch: {0}, Triplet Loss: {1}, finetune Accuracy: {2} \n".format(epoch, avg_triplet_loss, np.mean(accuracy)))


    if epoch >= 50 and epoch % 5 == 0:
        plot_scroc(fpr, tpr, figure_name='./log/tune_v2_roc_valid_epoch_{}.png'.format(epoch))
        torch.save(net, "./check_points/tune_v2_{}_checkpoint.pkl".format(epoch))
        print("save tune_v2_{}_checkpoint.pkl".format(epoch))

for i in range(total_epoch):
    train(i)



