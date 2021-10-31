import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss = torch.nn.BCEWithLogitsLoss()
    train_data = load_detection_data('dense_data/train', transform=dense_transforms.Compose([dense_transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor(), dense_transforms.ToHeatmap()]))
    valid_data = load_detection_data('dense_data/valid', transform=dense_transforms.Compose([dense_transforms.ToTensor(), dense_transforms.ToHeatmap()]))
    loss.to(device)
    global_step = 0
    for epoch in range(args.num_epoch):
        print("epoch #" + str(epoch))
        print("Training...")
        model.train()
        loss_vals = []
        for im, hm, _ in train_data:
            im, hm = im.to(device), hm.to(device)
            pred = model(im)
            loss_val = loss(pred, hm).mean()

            loss_vals.append(loss_val.detach().cpu().numpy())
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            #log(train_logger, im, hm, pred, global_step)

        avg_loss = sum(loss_vals) / len(loss_vals)
        model.eval()
        print("Validating...")
        for im, hm, _ in valid_data:
            im, hm = im.to(device), hm.to(device)
            #log(valid_logger, im, hm, model(im), global_step)
            
        print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))
    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=25)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-3)
    parser.add_argument('-mo', '--momentum', type=float, default=.9)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-6)  

    args = parser.parse_args()
    train(args)
