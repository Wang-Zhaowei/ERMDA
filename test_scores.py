# -*- coding: utf-8 -*-
from sklearn.metrics import roc_auc_score
import numpy as np


def calculate_performace(num, y_pred, y_prob, y_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(num):
        if y_test[index] ==1:
            if y_test[index] == y_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_test[index] == y_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    acc = float(tp + tn)/num
    try:
        precision = float(tp)/(tp + fp)
        recall = float(tp)/ (tp + fn)
        f1_score = float((2*precision*recall)/(precision+recall))
        #MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    except ZeroDivisionError:
        print("You can't divide by 0.")
        precision=recall=f1_score = 100
    AUC = roc_auc_score(y_test, y_prob)

    return tp, fp, tn, fn, acc, precision, recall, f1_score, AUC


def base_learners_results(metric_dict, fold_num, group_num, f):
    for i in range(group_num):
        ave_acc = 0
        ave_prec = 0
        ave_recall = 0
        ave_f1_score = 0
        ave_auc = 0
        ave_sum = 0
        bl_metric_list = []
        for fold in range(fold_num):
            temp_list = metric_dict[fold]
            bl_metric_list.append(temp_list[i])
        bl_metric_list = np.array(bl_metric_list)
        ave_acc = np.mean(bl_metric_list[:,0])
        ave_prec = np.mean(bl_metric_list[:,1])
        ave_recall = np.mean(bl_metric_list[:,2])
        ave_f1_score = np.mean(bl_metric_list[:,3])
        ave_auc = np.mean(bl_metric_list[:,4])
        ave_sum = np.mean(bl_metric_list[:,5])
        f.write('the '+ str(i+1)+ ' base learner proformance: \tAcc\t'+ str(ave_acc)+'\tprec\t'+ str(ave_prec)+ '\trecall\t'+str(ave_recall)+'\tf1_score\t'+str(ave_f1_score)+'\tAUC\t'+ str(ave_auc)+'\tSum\t'+ str(ave_sum)+'\n')