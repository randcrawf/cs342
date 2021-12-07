import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot


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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    loss = torch.nn.CrossEntropyLoss()
    train_data = SpeechDataset('data/train.txt', transform=one_hot);
    valid_data = SpeechDataset('data/valid.txt', transform=one_hot);
    for epoch in range(args.num_epoch):
        print("epoch: ", epoch)
        model.train()

        for im, gold in train_data:
            im, gold = im.to(device), gold.to(device)

            logit = model(im)
            loss_val = loss(logit, gold)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step() 


        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-3)
    parser.add_argument('-mo', '--momentum', type=float, default=.9)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5)

    args = parser.parse_args()
    train(args)
