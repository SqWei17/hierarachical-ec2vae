import json
import random
import torch
import os
import numpy as np
import time
from torch import nn
from struacture import Musicmodel
#from model import VAEbar2, VAEbar4
# from data_loader import MusicArrayLoader
from torch.autograd import Variable
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
# from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from IPython.display import Image, Audio, display, clear_output
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils import data

import csv

import codecs

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




# some initialization
with open('/home/model_config.json') as f:
    args = json.load(f)
if not os.path.isdir('log'):
    os.mkdir('log')
load_path = '/home/sqwei/ec4vae/model_parameters.pt'
# writer = SummaryWriter('log/{}'.format(args['modelname']))

print('Project initialized.', flush=True)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

model = Musicmodel(130, args['hidden_dim'], 3, 128,
                   128, 64, 128)
# musicvae = Musicmodel(130, args['hidden_dim'], 3, 12, args['pitch_dim'],args['rhythm_dim'], 32,128)


if args['if_parallel']:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
optimizer1 = optim.Adam(model.parameters(), lr=args['lr'])
# optimizer2 = optim.Adam(mlp.parameters(), lr=args['lr'])
if args['decay'] > 0:
    scheduler = MinExponentialLR(optimizer1, gamma=args['decay'], minimum=1e-5)
if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')
if os.path.exists(load_path):
    checkpoint = torch.load(load_path)
    model_dict = model.state_dict()
    for name in list(checkpoint.keys()):
        print(name)
        checkpoint[name.replace('module.', 'module.int_model4.')] = checkpoint.pop(name)
    state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
    for k,v in checkpoint.items():
        for name, p in model.named_parameters():
            if k ==name:
                print(k)
                p.requires_grad = False
    print(state_dict.keys()) # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    for p in model.parameters():
        print(p.requires_grad)
    print('load成功')
else:
    step, pre_epoch = 0, 0
    print('无保存模型，将从头开始训练！')

dataset = np.load("/home/data.npy", allow_pickle=True)
melody = dataset[0]
chord = dataset[1]
melody,chord = shuffle(melody,chord)
n_number = int(0.9 * melody.shape[0])


step_train = 0
pre_epoch = 0
train_data = Dataset_val(melody[:n_number], chord[:n_number])
train_loader = data.DataLoader(train_data, batch_size=256, shuffle=True)





def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")


def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function(recon0, recon_rhythm,
                  target_tensor1, rhythm_target,
                  distribution_01, distribution_11,
                  trip_lossp1, trip_lossp2, trip_lossp3, trip_lossr1, trip_lossr2, trip_lossr3,
                  beta=.1):
    CE1 = F.nll_loss(
        recon0.view(-1, recon0.size(-1)),
        target_tensor1,
        reduction='elementwise_mean')
    CE2 = F.nll_loss(
        recon_rhythm.view(-1, recon_rhythm.size(-1)),
        rhythm_target,
        reduction='elementwise_mean')

    normal1 = std_normal(distribution_01.mean.size())
    normal2 = std_normal(distribution_11.mean.size())
    KLD1 = kl_divergence(distribution_01, normal1).mean()
    KLD2 = kl_divergence(distribution_11, normal2).mean()

    max_indices = recon0.view(-1, recon0.size(-1)).max(-1)[-1]
    correct = max_indices == target_tensor1
    acc = torch.sum(correct.float()) / target_tensor1.size(0)

    max_indices2 = recon_rhythm.view(-1, recon_rhythm.size(-1)).max(-1)[-1]
    correct2 = max_indices2 == rhythm_target
    acc2 = torch.sum(correct2.float()) / rhythm_target.size(0)

    CPClossp1 = trip_lossp1.mean()  ###loss钟加入cpc
    CPClossp2 = trip_lossp2.mean()
    CPClossp3 = trip_lossp3.mean()
    CPClossr1 = trip_lossr1.mean()  ###loss钟加入cpc
    CPClossr2 = trip_lossr2.mean()
    CPClossr3 = trip_lossr3.mean()

    loss = CE1 + CE2 + beta * (KLD1 + KLD2) + CPClossp1 + CPClossp2 + CPClossp3 + CPClossr1 + CPClossr2 + CPClossr3
    vloss = CE1 + CE2 + beta * (KLD1 + KLD2)
    print(CPClossp1, CPClossp2)
    print(CPClossr1, CPClossr2)
    print(CPClossp3, CPClossr3)
    print(acc, acc2)
    print(vloss)
    print(CE1,CE2)
    print(KLD1,KLD2)
    return loss, CPClossp1, CPClossp2, CPClossp3, CPClossr1, CPClossr2, CPClossr3, vloss, acc, acc2, KLD1, KLD2


def train(train_loader, model, optimizer1, epoch):
    # batch, c = dl.get_batch(args['batch_size'])
    # batch = batch.numpy()
    # c = c.numpy()
    batch_loss = []
    batch_vae = []
    accc = []
    raccc = []
    cpcp1 = []
    cpcp2 = []
    cpcp3 = []
    cpcr1 = []
    cpcr2 = []
    cpcr3 = []
    kl1 = []
    kl2 = []
    model.train()
    epoch_loss = 0.
    epoch_ploss = epoch_rloss = epoch_bloss = 0.
    num_batch = len(train_loader)

    for i, batch in enumerate(train_loader):
        print(epoch)
        mel, ch = batch
        mel1 = mel.numpy()
        ch1 = ch.numpy()
        encode_tensor = torch.from_numpy(mel1).float()
        c = torch.from_numpy(ch1[:, :, :]).float()
        # et = encode_tensor.contiguous()
        # target_tensor =  et.view(-1, et.size(-1)).max(-1)[1]
        et1 = encode_tensor.contiguous()
        target_tensor = et1.view(-1, et1.size(-1)).max(-1)[1]

        rhythm_target = np.expand_dims(encode_tensor[:, :, :-2].sum(-1), -1)
        rhythm_target = np.concatenate((rhythm_target, encode_tensor[:, :, -2:]), -1)
        rhythm_target = torch.from_numpy(rhythm_target).float()
        rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]

        if torch.cuda.is_available():
            encode_tensor = encode_tensor.cuda()
            target_tensor = target_tensor.cuda()
            c = c.cuda()
            rhythm_target = rhythm_target.cuda()
        optimizer1.zero_grad()
        recon0, reconr_0, dis1m, dis1s, dis2m, dis2s, pNCEloss1, pNCEloss2, pNCEloss3, rNCEloss1, rNCEloss2, rNCEloss3 = model(
            encode_tensor, c)
        distribution_01 = Normal(dis1m, dis1s)
        distribution_11 = Normal(dis2m, dis2s)

        # valid = Variable(FloatTensor(recon0.size(0), 1).fill_(1.0), requires_grad=False)
        # fake = Variable(FloatTensor(recon0.size(0), 1).fill_(0.0), requires_grad=False)

        sum_loss, CPCP1, CPCP2, CPCP3, CPCR1, CPCR2, CPCR3, bloss, acc_1, acc_2, k1, k2 = loss_function(recon0,
                                                                                                        reconr_0,
                                                                                                        target_tensor,
                                                                                                        rhythm_target,
                                                                                                        distribution_01,
                                                                                                        distribution_11,
                                                                                                        pNCEloss1,
                                                                                                        pNCEloss2,
                                                                                                        pNCEloss3,
                                                                                                        rNCEloss1,
                                                                                                        rNCEloss2,
                                                                                                        rNCEloss3,
                                                                                                        beta=args[
                                                                                                            'beta'])

        sum_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer1.step()

        batch_loss.append(sum_loss.item())
        cpcp1.append(CPCP1.item())
        cpcp2.append(CPCP2.item())
        cpcp3.append(CPCP3.item())
        cpcr1.append(CPCR1.item())
        cpcr2.append(CPCR2.item())
        cpcr3.append(CPCR3.item())
        kl1.append(k1.item())
        kl2.append(k2.item())
        batch_vae.append(bloss.item())
        accc.append(acc_1.item())
        raccc.append(acc_2.item())

        if args['decay'] > 0:
            scheduler.step()
        # dl.shuffle_samples()
        array1 = np.array(batch_loss)
        array2 = np.array(cpcp1)
        array3 = np.array(cpcp2)
        array4 = np.array(cpcp3)
        array5 = np.array(cpcr1)
        array6 = np.array(cpcr2)
        array7 = np.array(cpcr3)
        array8 = np.array(batch_vae)
        array9 = np.array(accc)
        array10 = np.array(raccc)
        array11 = np.array(kl1)
        array12 = np.array(kl2)
    return (array1.mean(), array2.mean(), array3.mean(), array4.mean(), array5.mean(), array6.mean(), array7.mean(),
            array8.mean(), array9.mean(), array10.mean(), array11.mean(), array12.mean())




epoch = 0
lossData = [[]]
pcpc1 = [[]]
rcpc1 = [[]]
pcpc2 = [[]]
rcpc2 = [[]]
pcpc3 = [[]]
rcpc3 = [[]]
vaeloss = [[]]
tAcc = [[]]
rAcc = [[]]
kk1 = [[]]
kk2 = [[]]



best_valid_loss = float('inf')
train_ttloss, train_cpc1 = [], []
train_cpc2, train_vae = [], []
start = time.time()
while epoch < 100:
    print(f'Start Epoch: {epoch + 0:02}', flush=True)
    start_time = time.time()

    print(epoch)

    train_loss, CPCP1, CPCP2, CPCP3, CPCR1, CPCR2, CPCR3, basicloss, tacc, racc, kll1, kll2 = train(train_loader, model,
                                                                                                    optimizer1, epoch)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    pre_epoch = epoch
    state1 = {'model': model.cpu().state_dict(), 'optimizer': optimizer1.state_dict(), 'epoch': epoch}


    if epoch % 1 == 0:
        save_path1 = '/home/sqwei/ec4vae/param4inv4/train/model4.pt'
        # save_path1 = 'param/vaecpccos22/{}.pt'.format(epoch)
        torch.save(state1, save_path1)
        print("Model saved")

    if torch.cuda.is_available():
        model.cuda()
    print(f'Epoch: {epoch + 0:02} | Time: {epoch_mins}m {epoch_secs}s',
          flush=True)
    print(f'\tTrain Loss: {train_loss:.3f}', flush=True)

    lossData.append([epoch, train_loss])
    pcpc1.append([epoch, CPCP1])
    rcpc1.append([epoch, CPCR1])
    pcpc2.append([epoch, CPCP2])
    rcpc2.append([epoch, CPCR2])
    pcpc3.append([epoch, CPCP3])
    rcpc3.append([epoch, CPCR3])
    vaeloss.append([epoch, basicloss])
    tAcc.append([epoch, tacc])
    rAcc.append([epoch, racc])
    kk1.append([epoch, kll1])
    kk2.append([epoch, kll2])


    epoch = epoch + 1
