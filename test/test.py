# TODO:Evaluate test set accuracy
import time
import argparse
import logging
import os
import warnings
from torch.utils.data import DataLoader
import json
from utils.dataset import BasicDataset
from utils.eval import eval_net
from utils.logger import logger
from model.build_model.Build_model import *
from model.build_model.Build_swin import *

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
start_time = time.time()


def eval_dataset(n_img, json_path, class_json, runs, net, device, batch_size, img_scale):
    test_dataset = BasicDataset(n_img, json_path, img_scale, query='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logging.info('Test set:')
    miou = []
    mIoU, confusion, nc = eval_net(net, test_loader, device, class_json, n_img)
    miou.append(mIoU)
    confusion.plot(runs + "Confusion matrix.png", n_classes=nc)


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate test set', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_imgs', type=int, default=1, help='Number of input images')
    parser.add_argument('--json_path', type=str, default=r"F:/BS/Code/json/test.json", help='.json path')
    parser.add_argument('--net_type', type=str, default='mine', help='Net type')
    parser.add_argument('--batchsize', type=int, default=1, help='Number of batch size')
    parser.add_argument('--gpu_id', '-g', metavar='G', type=int, default=0, help='Number of gpu')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Scale factor for the input images')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    n_img = args.n_imgs
    json_path = args.json_path
    net_type = args.net_type
    batchsize = args.batchsize
    gpu_id = args.gpu_id
    scale = args.scale
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    assert os.path.exists(json_path), f"cannot find {json_path} file!"
    json_dict = json.load(open(json_path, 'r'))
    runs = json_dict['runs']
    class_json = json_dict['classes']
    n_classes = len(json.load(open(class_json, 'r')))
    ckpt = json_dict['ckpt']
    logger(net_type, runs)

    # Change here to adapt to the data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if net_type == 'mine':
        net = Build_model(n_classes=n_classes, use_uper=True)
    elif net_type == 'swin':
        net = Build_swin(n_classes=n_classes)
    else:
        raise NotImplementedError(f"net type:'{net_type}' does not exist, please check the 'net_type' arg!")

    logging.info(f'Loading model from {ckpt + net_type}.pth\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(ckpt + f"{net_type}.pth", map_location=device))
    eval_dataset(n_img, json_path, class_json, runs, net, device, batchsize, scale)
    logging.info(f'The entire {net_type} model test time is: {round((time.time() - start_time) / 3600, 2)} hours.')
