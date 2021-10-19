import torch
import numpy as np

from .models import FCN, save_model, ClassificationLoss
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # if args.continue_training:
    #     from os import path
    #     model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))


    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    loss = ClassificationLoss()
    print("Loading data...")
    train_data = load_dense_data('dense_data/train')
    valid_data = load_dense_data('dense_data/valid')
    torch.autograd.set_detect_anomaly(True)
    loss.to(device)
    global_step = 0
    for epoch in range(args.num_epoch):
        print("epoch #" + str(epoch))
        print("Training...")
        model.train()
        loss_vals, acc_vals, vacc_vals = [], [], []
        for im, label in train_data:
            im, label= im.to(device), label.to(device)
            pred = model(im)
            loss_val = loss(pred, label)
            acc_val = (model(im).argmax(1) == label).float().mean().item()

            loss_vals.append(loss_val.detach().cpu().numpy())
            acc_vals.append(acc_val)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        avg_loss = sum(loss_vals) / len(loss_vals)
        avg_acc = sum(acc_vals) / len(acc_vals)
        train_logger.add_scalar('accuracy', avg_acc, global_step)
        model.eval()
        print("Validating...")
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            vacc_vals.append((model(img).argmax(1) == label).float().mean().item())
            

        avg_vacc = sum(vacc_vals) / len(vacc_vals)
        valid_logger.add_scalar('accuracy', avg_vacc, global_step)
        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc))


    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

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
