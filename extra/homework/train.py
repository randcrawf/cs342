import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot
import numpy as np

def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = TCN().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your code from prior assignments
    Hint: SGD might need a fairly high learning rate to work well here

    """
    seq_len = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    loss = torch.nn.CrossEntropyLoss()

    print("Loading data...")
    train_data = SpeechDataset('data/train.txt', transform=one_hot);
    valid_data = SpeechDataset('data/valid.txt', transform=one_hot);

    def make_random_batch(batch_size, seq_len, is_train_data=True):
        B = []
        data = train_data if is_train_data else valid_data
        for i in range(batch_size):
            B.append(data[np.random.randint(0, len(data) - 1)][:,:])
        return torch.stack(B, dim=0)

    global_step = 0
    for epoch in range(args.num_epoch):
        print("epoch #" + str(epoch))
        print("Training...")
        model.train()
        loss_vals, valid_loss_vals = [], []
        batch = make_random_batch(args.batch_size, seq_len)
        batch_data = batch[:, :, :-1].to(device)
        batch_label = batch.argmax(dim=1).to(device)
        o = model(batch_data)
        loss_val = loss(o, batch_label)

        loss_vals.append(loss_val.detach().cpu().numpy())
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        global_step += 1
        
        avg_loss = sum(loss_vals) / len(loss_vals)
        train_logger.add_scalar('loss', avg_loss, global_step)

        model.eval()
        print("Validating...")
        valid_batches = make_random_batch(args.batch_size, seq_len+1, is_train_data=False)
        valid_batch_data = valid_batches[:, :, :-1].to(device)
        valid_batch_label = valid_batches.argmax(dim=1).to(device)
        valid_o = model(valid_batch_data)
        valid_loss_val = loss(valid_o, valid_batch_label)

        valid_loss_vals.append(valid_loss_val.detach().cpu().numpy())
        avg_valid_loss = sum(valid_loss_vals) / len(valid_loss_vals)
        valid_logger.add_scalar('loss', avg_valid_loss, global_step)
        print('epoch %-3d \t loss = %0.3f \t val loss = %0.3f' % (epoch, avg_loss, avg_valid_loss))
        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=10000)
    parser.add_argument('-lr', '--learning_rate', type=float, default=.1)
    parser.add_argument('-mo', '--momentum', type=float, default=.9)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('-bs', '--batch_size', type=float, default=128)

    args = parser.parse_args()
    train(args)
