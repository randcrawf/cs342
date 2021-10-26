import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    # from os import path
    # model = Detector()
    # train_logger, valid_logger = None, None
    # if args.log_dir is not None:
    #     train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
    #     valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # """
    # Your code here, modify your HW3 code
    # Hint: Use the log function below to debug and visualize your model
    # """
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # loss = torch.nn.BCEWithLogitsLoss()
    # train_data = load_detection_data('dense_data/train', num_workers=4)
    # valid_data = load_detection_data('dense_data/valid', num_workers=4)
    # loss.to(device)
    # global_step = 0
    # for epoch in range(args.num_epoch):
    #     print("epoch #" + str(epoch))
    #     print("Training...")
    #     model.train()
    #     loss_vals, acc_vals, vacc_vals = [], [], []
    #     for im, hm, _ in train_data:
    #         im, hm = im.to(device), hm.to(device)
    #         pred = model(im)
    #         loss_val = loss(pred, hm).mean()
    #         acc_val = (pred.argmax(1) == hm).float().mean().item()

    #         loss_vals.append(loss_val.detach().cpu().numpy())
    #         acc_vals.append(acc_val)
    #         optimizer.zero_grad()
    #         loss_val.backward()
    #         optimizer.step()
    #         global_step += 1
    #         log(train_logger, im, hm, pred, global_step)

    #     avg_loss = sum(loss_vals) / len(loss_vals)
    #     avg_acc = sum(acc_vals) / len(acc_vals)
    #     model.eval()
    #     print("Validating...")
    #     for im, hm, _ in valid_data:
    #         im, hm = im.to(device), hm.to(device)
    #         vacc_vals.append((model(im).argmax(1) == hm).float().mean().item())
    #         log(train_logger, im, hm, pred, global_step)
            

    #     avg_vacc = sum(vacc_vals) / len(vacc_vals)
    #     valid_logger.add_scalar('accuracy', avg_vacc, global_step)
    #     print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f \t iou = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc))
    # save_model(model)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Detector().to(device)

    """
    Your code here, modify your HW3 code
    """
    loss = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-6)

    train_data = load_detection_data('dense_data/train', num_workers=4)

    for epoch in range(50):
        print(epoch)
        model.train()
        for image, heatmap in train_data:
            image = image.to(device)
            heatmap = heatmap.to(device)

            pred_heatmap = model(image)

            l = loss(pred_heatmap, heatmap).mean()
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
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
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-mo', '--momentum', type=float, default=.9)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)  

    args = parser.parse_args()
    train(args)
