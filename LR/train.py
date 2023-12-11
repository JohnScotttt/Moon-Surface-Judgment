import sys
sys.path.append('..')

from Jtools import *

def train():
    batchsize = 512
    train_data = dlr.txtDataset('../data/txt_data/train.txt')
    val_data = dlr.txtDataset('../data/txt_data/val.txt')
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=len(val_data), shuffle=True)

    model = LRNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60], 0.1)
    loss_func = nn.BCELoss()
    epochs = 100
    best_f1 = 0

    for epoch in range(epochs):
        # training-----------------------------------
        model.train()
        train_loss = 0
        score = Score()
        for batch, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y.float().unsqueeze(1))
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            score.sum(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_acc, train_pre, train_rec, train_f1 = score.calculate()
        print(f'epoch:{epoch + 1}/{epochs}, train_loss:{train_loss/(math.ceil(len(train_data) / batchsize)):.4f}, '
              f'train_acc:{train_acc:.4f}, train_pre:{train_pre:.4f}, train_rec:{train_rec:.4f}, train_f1:{train_f1:.4f}')

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        score = Score()
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y.float().unsqueeze(1))
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            score.sum(pred, batch_y)
            val_acc, val_pre, val_rec, val_f1 = score.calculate()
            print(f'epoch:{epoch + 1}/{epochs}, val_loss:{eval_loss:.4f}, val_acc:{val_acc:.4f}, '
                  f'val_pre:{val_pre:.4f}, val_rec:{val_rec:.4f}, val_f1:{val_f1:.4f}')
        if val_f1 > best_f1:
            best_f1 = val_f1
            confirm('output')
            save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')

if __name__ == '__main__':
    train()