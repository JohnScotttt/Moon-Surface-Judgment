import sklearn
from net import LRNet
import torch
import dataloader as dlr
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import math
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train():
    batchsize = 512*512
    train_data = dlr.txtDataset('../data/txt_data/train.txt')
    val_data = dlr.txtDataset('../data/txt_data/val.txt')
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=len(val_data), shuffle=True)

    model = LRNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([1/9, 1]))
    epochs = 20
    for epoch in range(epochs):
        # training-----------------------------------
        model.train()
        train_loss = 0
        TTP = 0
        TTN = 0
        TFN = 0
        TFP = 0
        for batch, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            TP = ((pred == 1) & (batch_y == 1)).sum().item()
            TTP += TP
            TN = ((pred == 0) & (batch_y == 0)).sum().item()
            TTN += TN
            FN = ((pred == 0) & (batch_y == 1)).sum().item()
            TFN += FN
            FP = ((pred == 1) & (batch_y == 0)).sum().item()
            TFP += FP
            # acc = (TP + TN) / (TP + TN + FP + FN)
            # pre = TP / (TP + FP)
            # rec = TP / (TP + FN)
            # f1 = 2 * pre * rec / (pre + rec)
            # print(f'epoch:{epoch + 1}/{epochs}, batch:{batch + 1}/{math.ceil(len(train_data) / batchsize)}, loss:{loss.item():.4f}, acc:{acc:.4f}, pre:{pre:.4f}, rec:{rec:.4f}, f1:{f1:.4f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_acc = (TTP + TTN) / (TTP + TTN + TFP + TFN)
        train_pre = TTP / (TTP + TFP)
        train_rec = TTP / (TTP + TFN)
        train_f1 = 2 * train_pre * train_rec / (train_pre + train_rec)
        print(f'epoch:{epoch + 1}/{epochs}, train_loss:{train_loss/(math.ceil(len(train_data) / batchsize)):.4f}, train_acc:{train_acc:.4f}, train_pre:{train_pre:.4f}, train_rec:{train_rec:.4f}, train_f1:{train_f1:.4f}')

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            val_acc = accuracy_score(batch_y, pred)
            val_pre = precision_score(batch_y, pred)
            val_rec = recall_score(batch_y, pred)
            val_f1 = f1_score(batch_y, pred)
            print(f'epoch:{epoch + 1}/{epochs}, val_loss:{eval_loss:.4f}, val_acc:{val_acc:.4f}, val_pre:{val_pre:.4f}, val_rec:{val_rec:.4f}, val_f1:{val_f1:.4f}')
        # torch.save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')

if __name__ == '__main__':
    train()