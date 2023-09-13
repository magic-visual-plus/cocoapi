import pytest
from loguru import logger 

import os
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path

def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    # y = smooth(py.mean(0), 0.05)
    # ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)

metric="bbox"

eps=1e-16
cur_dir = os.path.dirname(os.path.abspath(__file__))

coco_gt = COCO(f'{cur_dir}/gt_annotations.coco.json')
coco_dt = coco_gt.loadRes(f'{cur_dir}/predictions.json')

coco_eval = COCOeval(coco_gt, coco_dt, metric)
coco_eval.evaluate()
coco_eval.accumulate_details()
# coco_eval.summarize()
coco_eval.summarize_details()

px = np.linspace(0, 1, 101)

'''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
'''

re_precisions = coco_eval.eval["re_precision"]
re_precision_50 = re_precisions[0, :, 0, 0, 2]
print(re_precision_50.shape)

plot_mc_curve(px, re_precision_50[None, :], save_dir=Path('precision_50.png'), names=['y'], ylabel='precision')
re_recalls = coco_eval.eval["re_recall"]
re_recall_50 = re_recalls[0, :, 0, 0, 2]
print(re_recall_50.shape)
plot_mc_curve(px, re_recall_50[None, :], save_dir=Path('recall_50.png'), names=['y'], ylabel='recall')

p = re_precision_50[None, :]
r = re_recall_50[None, :]

f1 = 2 * p * r / (p + r + eps)
i = f1.mean(0).argmax()  # max F1 index
p, r, f1 = p[:, i], r[:, i], f1[:, i]
confidence = px[i]
print(confidence)
logger.info("best precison {}, recall {}, confidence {}, f1 {}", p, r, confidence, f1)
# extract eval data
if False:
    precisions = coco_eval.eval["precision"]
    '''
        precisions[T, R, K, A, M]
        T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
        R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
        K: category, idx from 0 to ...
        A: area range, (all, small, medium, large), idx from 0 to 3
        M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    pr_array1 = precisions[0, :, 0, 0, 2] 
    pr_array2 = precisions[1, :, 0, 0, 2] 
    pr_array3 = precisions[2, :, 0, 0, 2] 
    pr_array4 = precisions[3, :, 0, 0, 2] 
    pr_array5 = precisions[4, :, 0, 0, 2] 
    pr_array6 = precisions[5, :, 0, 0, 2] 
    pr_array7 = precisions[6, :, 0, 0, 2] 
    pr_array8 = precisions[7, :, 0, 0, 2] 
    pr_array9 = precisions[8, :, 0, 0, 2] 
    pr_array10 = precisions[9, :, 0, 0, 2] 
    print(f'pr_array9 shape {pr_array9.shape}')

    x = np.arange(0.0, 1.01, 0.01)
    # plot PR curve
    plt.plot(x, pr_array1, label="iou=0.5")
    plt.plot(x, pr_array2, label="iou=0.55")
    plt.plot(x, pr_array3, label="iou=0.6")
    plt.plot(x, pr_array4, label="iou=0.65")
    plt.plot(x, pr_array5, label="iou=0.7")
    plt.plot(x, pr_array6, label="iou=0.75")
    plt.plot(x, pr_array7, label="iou=0.8")
    plt.plot(x, pr_array8, label="iou=0.85")
    plt.plot(x, pr_array9, label="iou=0.9")
    plt.plot(x, pr_array10, label="iou=0.95")

    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.show()
    plt.savefig("pr_curve.png")
