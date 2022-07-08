 ## seg + class

import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.unet import Unet
from nets.unet_training import CE_Loss, Dice_loss, LossHistory_seg, Focal_Loss
from utils.dataloader_all import UnetDataset,unet_dataset_collate
from utils.metrics import f_score, results
from utils.fit_MTL import fit_one_epoch_MTL

import torch.backends.cudnn as cudnn
# from nets import get_model_from_name
from utils.callbacks import LossHistory, AccHistory, f_score, F_scoreHistory, SigmaHistory
from utils.dataloader_class import DataGenerator_class, detection_collate_class
from utils.utils import get_classes, weights_init
# from utils.utils_fit_class import fit_one_epoch_class
# from utils.utils_fit_total import fit_one_epoch
import torch.nn.functional as F
from torch import nn
from utils.AutomaticWeightedLoss import AutomaticWeightedLoss
from nets.vgg16 import vgg16
from nets.multi_task_net import multi_model_unet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device_ids = [0, 1]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    Cuda = True
    # ----------------------------------------------------#
    #   训练自己的数据集的时候一定要注意修改classes_path
    #   修改成自己对应的种类的txt
    # ----------------------------------------------------#
    classes_path = 'model_data/cls_classes.txt'
    input_shape = [512, 512]

    # -------------------------------#
    #   训练自己的数据集必须要修改的
    #   自己需要分割的分类个数+1，如2 + 1
    # -------------------------------#
    num_classes_seg = 2  #类别+背景


    pretrained = True
    # ------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------#
    Freeze_Train = True
    # ------------------------------------------------------#
    #   获得图片路径和标签
    # ------------------------------------------------------#
    annotation_path = "ori_small_cls_train.txt"
    logs_name_cla = 'adaLogs_class'
    logs_name_seg = 'adaLogs_seg'
    logs_sigma    = 'adaptively_sigma'

    # ------------------------------------------------------#
    #   进行训练集和验证集的划分，默认使用10%的数据用于验证
    # ------------------------------------------------------#
    val_split = 0.25
    # ------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，0代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------#
    num_workers = 2

    # ------------------------------------------------------#
    #   获取classes      classification
    # ------------------------------------------------------#
    class_names, num_classes_cla = get_classes(classes_path)


    loss_history_cla = LossHistory(logs_name_cla + "/")
    acc_history_cla = AccHistory(logs_name_cla + "/")
    ##
    ## print sigma result        ++
    sigma_history   = SigmaHistory(logs_sigma + "/")
    # ----------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    # ----------------------------------------------------#
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    dropout = 0.5
    ##-----------------------------------------------------------------------------------------#
    #                           seg
    # -----------------------------------------------------------------------------------------#

    VOCdevkit_path = "datasets_seg"
    dice_loss =True
    # ---------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    # ---------------------------------------------------------------------#
    focal_loss = False
    num_classes = 2

    cls_weights = np.ones([num_classes_seg], np.float32) # 不同类设置相等的权重
    # ------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------#
    Freeze_Train = True

    num_workers = 4 #   用于设置是否使用多线程读取数据
    npz_path = r'C:\Users\dell\PycharmProjects\tongue\MTL\fold_npz\fold1_data.npz'
    train_split = 0.75
    val_split = 1.00
    x = np.load(npz_path)
    names_all = x['name']
    labels_all = x['label']
    path_all = x['path']

    train_line = int(len(names_all) * train_split)
    val_line = int(len(names_all) * val_split)
    train_lines = names_all[:train_line]
    val_lines = names_all[train_line:val_line]
    names_train = names_all[:train_line]
    names_val = names_all[train_line:val_line]
    labels_train = labels_all[:train_line]
    labels_val = labels_all[train_line:val_line]

    model = multi_model_unet(num_classes=num_classes, pretrained=pretrained, p=dropout).train()

    if not pretrained:
        weights_init(model)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss_history_seg = LossHistory(logs_name_seg + "/")
    f_score_history = F_scoreHistory(logs_name_seg + "/")

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(os.path.join(VOCdevkit_path, "ImageSets_all/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()

    with open(os.path.join(VOCdevkit_path, "ImageSets_all/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    if True:
        # ----------------------------------------------------#
        #   冻结阶段训练参数
        #   此时模型的主干被冻结了，特征提取网络不发生改变
        #   占用的显存较小，仅对网络进行微调
        # ----------------------------------------------------#
        lr = 1e-3
        batch_size = 5
        Init_Epoch = 0
        Freeze_Epoch = 50
        UnFreeze_Epoch = 100
        Epoch = UnFreeze_Epoch - Init_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # optimizer = optim.Adam(model_train_cla.parameters(), lr, weight_decay=5e-4)
        # 传入两个模型：
        awl = AutomaticWeightedLoss(2)  # we have 2 losses
        optimizer = torch.optim.Adam([
            {'params': model_train.parameters(), 'lr': lr, 'weight_decay': 5e-4}, #大括号里的信息使用字典形式
            {'params': awl.parameters(), 'weight_decay': 0}
        ])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
        ## print train learning rate
        # lr_renew = lr_scheduler.get_last_lr()
        train_dataset = UnetDataset(train_lines, input_shape, num_classes, names_train, labels_train, True,
                                    VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, names_val, labels_val, False, VOCdevkit_path)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate)

        # if Freeze_Train:
        #     model_cla.Unfreeze_backbone()
        #     model_seg.unfreeze_backbone()


        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch_MTL(model_train, model, sigma_history,loss_history_seg, f_score_history,
                              gen, gen_val, dice_loss, focal_loss, cls_weights, num_classes_cla,
                              loss_history_cla, acc_history_cla,
                              Epoch, epoch, epoch_step, epoch_step_val, optimizer, Cuda, logs_name_cla, logs_name_seg)
            lr_scheduler.step()
