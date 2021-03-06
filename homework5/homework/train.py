from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Planner().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss = torch.nn.L1Loss().to(device)

    transform = dense_transforms.Compose([dense_transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor()])
    train_data = load_data('drive_data',transform=transform, num_workers=4)

    for epoch in range(args.num_epoch):
        print("epoch: ", epoch)
        model.train()

        for im, label in train_data:
            im, label = im.to(device), label.to(device)

            logit = model(im)
            loss_val = loss(logit, label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step() 


        save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

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
