import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import time
from .utils import get_lr
from nets.unet_training import CE_Loss, Dice_loss, LossHistory_seg, Focal_Loss
from utils.callbacks import LossHistory, AccHistory, f_score, F_scoreHistory, SigmaHistory,seg_metric
from utils.AutomaticWeightedLoss import AutomaticWeightedLoss
from utils.metrics import f_score, results

def fit_one_epoch_MTL(model_train, model, sigma_history,loss_history_seg, f_score_history,
                              gen, gen_val, dice_loss, focal_loss, cls_weights, num_classes,
                              loss_history_cla, acc_history_cla,
                              Epoch, epoch, epoch_step, epoch_size_val, optimizer, Cuda, logs_name_cla, logs_name_seg):
    ####  seg  ##########################
    model_train.train()
    total_loss_seg = 0
    total_f_score = 0
    val_loss_seg = 0
    val_f_score = 0
    val_DSC = 0
    val_JI = 0
    ###########################################
    ## class  ##########--------------------------------------------------
    total_loss_class = 0
    total_accuracy_class = 0
    val_loss_class = 0
    val_accuracy_class = 0
    val_precision_class = 0
    val_recall_class = 0
    val_f1_class = 0
    sig1_result = 0
    sig2_result = 0

    ################################################
    start_time = time.time()
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, seg_labels, y_labels = batch

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                seg_labels = torch.from_numpy(seg_labels).type(torch.FloatTensor)
                y_labels = torch.from_numpy(y_labels).type(torch.FloatTensor).long()
                weights = torch.from_numpy(cls_weights)
                if Cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    seg_labels = seg_labels.cuda()
                    y_labels = y_labels.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            result = model_train(imgs)  # 同时输出两个任务的指标

            outputs_seg = result['mean_seg']

            outputs_cla = result['mean_cla']

            if focal_loss:
                loss_seg1 = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_classes)
            else:
                loss_seg1 = CE_Loss(outputs_seg, pngs, weights, num_classes=num_classes)
            ## cla---------------------------------------
            # outputs_cla = model_train_cla(images)
            loss_class1 = nn.CrossEntropyLoss()(outputs_cla, y_labels)
            if dice_loss:
                main_dice = Dice_loss(outputs_seg, seg_labels)
                loss_seg1 = loss_seg1 + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs_seg, y_labels)

            ## totally_loss ------------------------------------------------------------------
            ## equal weight
            # totally_loss = loss_seg1 + loss_class1
            # uncertainty weight

            loss1 = loss_seg1
            loss2 = loss_class1
            ###   --------------------------------------------------------------
            awl = AutomaticWeightedLoss(2)
            totally_loss, sig1, sig2 = awl(loss1, loss2)
            # totally_loss =
            totally_loss.backward()
            optimizer.step()

            total_loss_seg += loss_seg1.item()
            total_loss_class += loss_class1.item()
            total_f_score += _f_score.item()
            # ## 收集权重大小的变化
            sig1_result += sig1.item()
            sig2_result += sig2.item()

            with torch.no_grad():
                # -------------------------------#
                #   计算accuracy
                # -------------------------------#
                accuracy = torch.mean(
                    (torch.argmax(F.softmax(outputs_cla, dim=-1), dim=-1) == y_labels).type(torch.FloatTensor))

                total_accuracy_class += accuracy.item()
            ## 训练集输出结果：
            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss_seg': total_loss_seg / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'sig1'   : sig1_result / (iteration + 1),
                                'sig2'   : sig2_result / (iteration + 1),
                                's/step': waste_time,
                                'lr': get_lr(optimizer)})
            pbar.update(1)
            pbar.set_postfix(**{'total_loss_cla': total_loss_class / (iteration + 1),
                                'accuracy': total_accuracy_class / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
            ## 收集权重大小的变化
    sigma_history.append_sigma(sig1_result / (epoch + 1), sig2_result / (epoch + 1))



            # start_time = time.time()
    print('finish Train')

    model_train.train()

    ##---------  Start Validation   --------------------------------------------------------
    print('Start Validation')
    with tqdm(total=Epoch, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, (batch_seg, batch_cla) in enumerate(gen_val):
            if iteration >= Epoch:
                break

            imgs, pngs, labels = batch_seg
            images, targets = batch_cla

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).type(torch.FloatTensor).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                images = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor).long()
                if Cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()
                    images = images.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()

                outputs_seg = model_train(imgs)

                # ----------------  seg  -------------------#
                if focal_loss:
                    loss_seg1 = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_classes)
                else:
                    loss_seg1 = CE_Loss(outputs_seg, pngs, weights, num_classes=num_classes)
                if dice_loss:
                    main_dice = Dice_loss(outputs_seg, labels)
                    loss_seg1 = loss_seg1 + main_dice
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs_seg, labels)

                seg_result = seg_metric(outputs_seg, seg_labels)

                # seg
                DSC = seg_result['DSC']
                val_DSC += DSC.item()
                val_loss_seg += loss_seg1.item()
                val_f_score += _f_score.item()

                #----------------  class  -------------------#
                val_loss_class1 = nn.CrossEntropyLoss()(outputs_cla, targets)

                val_loss_class += val_loss_class1.item()
                with torch.no_grad():
                    # accuracy = torch.mean(
                    #     (torch.argmax(F.softmax(outputs_cla, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                    outputs_1 = torch.argmax(F.softmax(outputs_cla, dim=-1), dim=-1)
                    accuracy, precision, recall, f1 = results(targets, outputs_1)
                    val_accuracy_class += accuracy.item()
                    val_precision_class += precision.item()
                    val_recall_class += recall.item()
                    val_f1_class += f1.item()
            pbar.set_postfix(**{'total_loss': val_loss_class / (iteration + 1),
                                'accuracy': val_accuracy_class / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'precision': val_precision_class / (iteration + 1),
                                'recall': val_recall_class / (iteration + 1),
                                'f1': val_f1_class / (iteration + 1)})
            pbar.update(1)

            pbar.set_postfix(**{'total_loss': val_loss_seg / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'dsc': val_DSC / (iteration + 1),
                                'JI' :val_JI / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    #--------------  seg  ------------------------------------#
    loss_history_seg.append_loss(total_loss_seg / (epoch + 1), val_loss_seg / (epoch_size_val + 1))
    f_score_history.append_f_score(total_f_score / (epoch + 1), val_f_score / (epoch_size_val + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss_seg / (epoch + 1), val_loss_seg / (epoch_size_val + 1)))
    print('Total f_score: %.4f || Val f_score: %.4f ' % (total_f_score / (epoch + 1), val_f_score / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch + 1))

    # --------------  class  ------------------------------------#
    loss_history_cla.append_loss(total_loss_class / epoch, val_loss_class / epoch_size_val)
    acc_history_cla.append_acc(total_accuracy_class / epoch, val_accuracy_class / epoch_size_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss_class / epoch, val_loss_class / epoch_size_val))
    print('Total Acc: %.3f || Val Acc: %.3f ' % (total_accuracy_class / epoch, val_accuracy_class / epoch_size_val))
    print('Val Acc: %.3f ' % (val_accuracy_class / epoch_size_val))
    print('Val precision: %.3f ' % (val_precision_class / epoch_size_val))
    print('Val recall: %.3f ' % (val_recall_class / epoch_size_val))
    print('Val f1: %.3f ' % (val_f1_class / epoch_size_val))


    print('Finish Validation')
