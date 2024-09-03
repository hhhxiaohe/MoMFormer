# TODO:Train model
import argparse
import json
import logging
import math
import os
import time
import warnings
import torch
import torch.optim.lr_scheduler as lr_scheduler
from apex import amp
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from utils.creat_model import creat_model
from utils.dataset import BasicDataset
from utils.eval import eval_net
from utils.logger import logger
from utils.plot_train import *

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
start_time = time.time()


def train_net(net_type, net, device, epochs, batch_size, warm_epochs, lr, weight_decay, img_scale=1, apex=True):
    global best_miou_last, i
    train_dataset = BasicDataset(n_img, json_path, img_scale, query='train')
    val_dataset = BasicDataset(n_img, json_path, img_scale, query='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 3, shuffle=False, num_workers=4, pin_memory=True)
    global_step = 0
    logging.info(f'''Using device {device}
        Starting training:
                     Net type:           {net_type}
                     Epochs:             {epochs}
                     Batch size:         {batch_size}
                     Learning rate:      {lr}
                     Input img number:   {n_img}
                     Input channels:     {in_chans}         
                     Dataset size:       {len(train_dataset) + len(val_dataset)}
                     Training size:      {len(train_dataset)}
                     Validation size:    {len(val_dataset)}
                     Device:             {device.type}
                     Apex:               {apex}
                     Images scaling:     {img_scale}\n''')

    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    if apex:
        model, optimizer = amp.initialize(net, optimizer, opt_level="O0")

    e = warm_epochs
    lf = lambda x: (((1 + math.cos((x - e + 1) * math.pi / (epochs - e))) / 2) ** 1.0) * 0.95 + 0.05 if x >= e else 0.95 / (e - 1) * x + 0.05
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=-1)
    scheduler.last_epoch = global_step
    lrs, losses, miou = [], [], []
    best_miou = 0

    for epoch in range(epochs):
        cur_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch= {epoch+1},  lr= {cur_lr}')
        net.train()
        epoch_loss = 0

        for i, batch in enumerate(train_loader):
            train_imgs = []
            for n in range(1, 1+n_img):
                imgs = batch[f'image{n}']
                imgs = imgs.to(device=device, dtype=torch.float32, non_blocking=True)
                train_imgs.append(imgs)

            true_masks = batch['mask']
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type, non_blocking=True)

            optimizer.zero_grad()
            masks_pred = net(train_imgs)
            train_imgs.clear() if isinstance(train_imgs, list) else None

            if net.n_classes > 1:
                if net.aux:
                    criterion_main = nn.CrossEntropyLoss()
                    criterion_aux = nn.CrossEntropyLoss()
                    loss_main = criterion_main(masks_pred[0], torch.squeeze(true_masks, dim=1))
                    loss_aux = criterion_aux(masks_pred[1], torch.squeeze(true_masks, dim=1))
                    loss = loss_main + loss_aux * 0.4
                else:
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(masks_pred, torch.squeeze(true_masks, dim=1))
            else:
                if net.aux:
                    criterion_main = nn.BCEWithLogitsLoss()
                    criterion_aux = nn.BCEWithLogitsLoss()
                    loss_main = criterion_main(masks_pred[0], true_masks)
                    loss_aux = criterion_aux(masks_pred[1], true_masks)
                    loss = loss_main + loss_aux * 0.4
                else:
                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(masks_pred, true_masks)
            epoch_loss += loss.item()

            if apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            global_step += 1

        torch.cuda.empty_cache()
        logging.info('Val set:')
        mIoU, confusion, nc = eval_net(net, val_loader, device, class_json, n_img)
        torch.cuda.empty_cache()
        miou.append(mIoU)

        epoch_loss_mean = epoch_loss / (i + 1)
        logging.info(f'Epoch loss: {epoch_loss_mean}')
        losses.append(epoch_loss_mean)

        scheduler.step()
        lrs.append(cur_lr)

        if mIoU > best_miou:
            try:
                os.mkdir(runs + "ckpts")
                logging.info('Created ckpt directory')
            except OSError:
                pass
            torch.save(net.state_dict(), runs + f"ckpts/{net_type}_best.pth")
            best_miou = mIoU
            logging.info(f'\t\t\t\tEpoch {epoch + 1} saved ! miou(val) = {mIoU}')
            confusion.plot(runs + "Confusion matrix.png", n_classes=nc)
        logging.info('The run time is: {} hours\n'.format(round((time.time() - start_time) / 3600, 3)))

        if epoch + 1 == epochs:
            logging.info('Computing training set accuracy...\nTrain set:')
            torch.cuda.empty_cache()
            eval_net(net, train_loader, device, class_json, n_img)
            torch.cuda.empty_cache()
            logging.info('The run time is: {} hours\n'.format(round((time.time() - start_time) / 3600, 3)))

    plot_miou(miou, runs)
    plot_loss(losses, runs)
    plot_lr(lrs, runs)


def get_args():
    parser = argparse.ArgumentParser(description='Train model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_chans', type=int and tuple, default=(3,3), help='Channels of input images')
    parser.add_argument('--img_size', type=int, default=512, help='Images size')
    parser.add_argument('--json_path', type=str, default=r"E:/python_doc/Code/json/train.json", help='.json path')
    parser.add_argument('--net_type', type=str, default='MoMFormer', help='Net type')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--warm_epochs', type=int, default=5, help='Number of warm epochs')
    parser.add_argument('--batchsize', type=int, default=2, help='Number of batch size')
    parser.add_argument('--lr', type=float, default=0.00004, help='Learning rate starting value')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='L2 regular term')
    parser.add_argument('--gpu_id', type=int, default=0, help='Number of gpu')
    parser.add_argument('--if_apex', default=False, help='Whether to enable Apex mixed precision training')
    parser.add_argument('--load', default=None, help='Load model from a .pth file')
    parser.add_argument('--scale', type=float, default=1, help='Images scale')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    json_path = args.json_path
    in_chans = args.in_chans
    n_img = len(in_chans) if isinstance(in_chans, tuple) else 1
    gpu_id = args.gpu_id
    net_type = args.net_type
    if_apex = args.if_apex
    scale = args.scale
    img_size = int(args.img_size * scale)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    assert os.path.exists(json_path), f"Cannot find {json_path} file!"
    json_dict = json.load(open(json_path, 'r'))
    runs = json_dict['runs']
    class_json = json_dict['classes']
    n_classes = len(json.load(open(class_json, 'r')))
    logger(net_type, runs)
    logging.info(f'Run path: {runs}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




    net = creat_model(model_type=net_type, n_classes=n_classes, in_chans=in_chans, img_size=img_size)



    logging.info(f'Network: {net.n_classes} output channels (classes)')
    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net.to(device=device, non_blocking=True)

    if args.load is not None:
        net.eval()
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    # faster convolutions, but more memory
    # cudnn.benchmark = True
    train_net(net_type=net_type, net=net, epochs=args.epochs, batch_size=args.batchsize, warm_epochs=args.warm_epochs,
              lr=args.lr, weight_decay=args.weight_decay, device=device, img_scale=scale, apex=if_apex)

    torch.save(net.state_dict(), runs + f"ckpts/{net_type}_last.pth")
    end_time = time.time()
    run_time = end_time - start_time
    logging.info('The entire {} model run time is: {} hours.'.format(net_type, round(run_time / 3600, 2)))
