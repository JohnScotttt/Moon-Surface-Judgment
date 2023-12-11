import sys
sys.path.append('..')

from Jtools import *

def train():
    batchsize = 16
    train_data = dlr.imgDataset('../data/DL_txt_data/train.txt', transform=transforms.ToTensor())
    val_data = dlr.imgDataset('../data/DL_txt_data/val.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=len(val_data), shuffle=True)

    model = UNet()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70, 140], 0.1)
    loss_func = nn.CrossEntropyLoss()
    epochs = 200
    best_f1 = 0
    # refresh('output')
    for epoch in range(epochs):
        # training-----------------------------------
        model.train()
        train_loss = 0
        score = Score()
        for batch_x, batch_y in tqdm(train_loader):
            batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            batch_y = batch_y.squeeze(1)
            out = model(batch_x)
            loss = loss_func(out, batch_y.to(dtype=torch.long))
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            score.sum(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_acc, train_pre, train_rec, train_f1, train_iou = score.calculate()
        print(f'epoch:{epoch + 1}/{epochs}, train_loss:{train_loss/(math.ceil(len(train_data) / batchsize)):.4f}, '
              f'train_acc:{train_acc:.4f}, train_pre:{train_pre:.4f}, train_rec:{train_rec:.4f}, train_f1:{train_f1:.4f}')
        loss_list.append(train_loss / (math.ceil(len(train_data) / batchsize)))
        acc_list.append(train_acc)
        pred_list.append(train_pre)
        rec_list.append(train_rec)
        f1_list.append(train_f1)
        iou_list.append(train_iou)

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        score = Score()
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            batch_y = batch_y.squeeze(1)
            out = model(batch_x)
            loss = loss_func(out, batch_y.to(dtype=torch.long))
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            score.sum(pred, batch_y)
            val_acc, val_pre, val_rec, val_f1, val_iou = score.calculate()
            print(f'epoch:{epoch + 1}/{epochs}, val_loss:{eval_loss:.4f}, val_acc:{val_acc:.4f}, '
                  f'val_pre:{val_pre:.4f}, val_rec:{val_rec:.4f}, val_f1:{val_f1:.4f}')
        # if best_f1 < val_f1:
        #     best_f1 = val_f1
        #     save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')

if __name__ == '__main__':
    loss_list = []
    acc_list = []
    pred_list = []
    rec_list = []
    f1_list = []
    iou_list = []
    train()
    plt.plot(loss_list, label='loss')
    plt.plot(acc_list, label='acc')
    plt.plot(pred_list, label='precision')
    plt.plot(rec_list, label='recall')
    plt.plot(f1_list, label='f1')
    plt.plot(iou_list, label='iou')
    plt.legend()
    plt.show()