from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # if args.continue_training:
    #     from os import path
    #     model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))


    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    loss = ClassificationLoss()
    print("Loading data...")
    train_data = load_data('data/train')
    valid_data = load_data('data/valid')
    torch.autograd.set_detect_anomaly(True)
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
            acc_val = accuracy(pred, label)

            loss_vals.append(loss_val.detach().cpu().numpy())
            acc_vals.append(acc_val.detach().cpu().numpy())
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
            vacc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
            

        avg_vacc = sum(vacc_vals) / len(vacc_vals)
        valid_logger.add_scalar('accuracy', avg_vacc, global_step)
        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc))


    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-mo', '--momentum', type=float, default=.9)
    args = parser.parse_args()
    train(args)
