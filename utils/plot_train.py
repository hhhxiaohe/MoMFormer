# TODO:Plot the training results
import matplotlib.pyplot as plt


def plot_miou(miou, runs):
    plt.figure(figsize=(9, 9), tight_layout=True)
    plt.plot([i + 1 for i in range(len(miou))], miou, '.-', label='mIoU')
    plt.tick_params(labelsize=14)
    plt.xlabel('Epoch', size=18, color='purple')
    plt.ylabel('mIoU', size=18, color='purple')
    plt.legend(loc=0)
    plt.grid(ls='--')
    plt.tight_layout()
    plt.savefig(runs + "miou_result.png", dpi=150)
    plt.close()


def plot_loss(loss, runs):
    plt.figure(figsize=(9, 9), tight_layout=True)
    plt.plot([i + 1 for i in range(len(loss))], loss, '.-', label='Loss')
    plt.tick_params(labelsize=14)
    plt.xlabel('Epoch', size=18, color='purple')
    plt.ylabel('Loss', size=18, color='purple')
    plt.legend(loc=0)
    plt.grid(ls='--')
    plt.tight_layout()
    plt.savefig(runs + "Loss.png", dpi=150)
    plt.close()


def plot_lr(lr, runs):
    plt.figure(figsize=(9, 9), tight_layout=True)
    plt.plot([i + 1 for i in range(len(lr))], lr, '.-', label='LambdaLR')
    plt.tick_params(labelsize=14)
    plt.xlabel('Epoch', size=18, color='purple')
    plt.ylabel('LR', size=18, color='purple')
    plt.legend(loc=0)
    plt.grid(ls='--')
    plt.tight_layout()
    plt.savefig(runs + "LR.png", dpi=150)
    plt.close()
