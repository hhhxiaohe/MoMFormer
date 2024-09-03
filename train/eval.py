# TODO:Evaluate model accuracy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import os
import json
import seaborn as sns
import logging


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        """update confusion matrix"""
        for p, t in zip(preds, labels):
            self.matrix[int(p), int(t)] += 1

    def summary(self):
        mIoU = 0
        table = PrettyTable()
        table.field_names = ["Per-class", "IoU (%)"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            IoU = TP / (TP + FP + FN) * 100 if TP + FP != 0 else 0.
            table.add_row([self.labels[i], round(IoU, 3)])
            mIoU += IoU
        logging.info(f'IoU:\n{table}')
        mIoU = round(mIoU / self.num_classes, 3)
        logging.info(f'\tmIoU: {mIoU}')
        return mIoU

    def plot(self, save_path, n_classes):
        """plot confusion matrix"""
        # matrix = self.matrix / (self.matrix.sum(0).reshape(1, n_classes) + 1E-8)  # normalize by classes
        matrix = self.matrix / self.matrix.sum()
        plt.figure(figsize=(12, 9), tight_layout=True)
        sns.set(font_scale=1.0 if n_classes < 50 else 0.8)
        sns.heatmap(data=matrix,
                    annot=n_classes < 30,
                    annot_kws={"size": 12},
                    cmap='Blues',
                    fmt='.3f',
                    square=True,
                    linewidths=.16,
                    linecolor='white',
                    xticklabels=self.labels,
                    yticklabels=self.labels)

        plt.xlabel('True', size=16, color='purple')
        plt.ylabel('Predicted', size=16, color='purple')
        plt.title('Normalized confusion matrix', size=20, color='blue')
        plt.savefig(save_path, dpi=150)
        plt.close()


def eval_net(net, loader, device, json_label_path, n_img):
    assert os.path.exists(json_label_path), 'Cannot find {} file'.format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for label, _ in class_indict.items()]
    n_classes = 2 if net.n_classes == 1 else net.n_classes
    confusion = ConfusionMatrix(num_classes=n_classes, labels=labels)
    net.eval()

    with torch.no_grad():
        for batch in loader:
            images = []
            for n in range(1, 1 + n_img):
                imgs = batch[f'image{n}']
                imgs = imgs.to(device=device, dtype=torch.float32)
                images.append(imgs)

            true_masks = batch['mask']
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            mask_pred = net(images)
            images.clear()

            for true_mask, pred in zip(true_masks, mask_pred):
                true_mask = true_mask.float()
                true_mask = true_mask.squeeze(dim=0).cpu().numpy().flatten()
                if net.n_classes > 1:
                    pred = F.softmax(pred, dim=0)
                    pred = torch.argmax(pred, dim=0)
                    pred = pred.cpu().numpy().flatten()
                else:
                    pred = (pred > 0.5).int().squeeze(dim=0).cpu().numpy().flatten()

                confusion.update(pred, true_mask)

        mIoU = confusion.summary()
    return mIoU, confusion, n_classes
