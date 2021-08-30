# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np


def base_rf_learners(group_num,local_time,hy1,hy2):
    clfs = []
    with open('./results/base learners parameter.txt','a') as f:
        f.write('------------- Base learners - '+ local_time +' ---------------\n')
        for i in range(group_num):
            print('RF'+str(i+1)+': n_estimators = '+str(hy1)+' max_depth: '+str(hy2)+'\n')
            f.write('RF'+str(i+1)+'\tn_estimators\t'+str(hy1)+'\tmax_depth\t'+str(hy2)+'\n')
            clfs.append(RandomForestClassifier(n_estimators=hy1,max_depth=hy2,n_jobs=-1))
        f.write('----------------------------------------------------------------------\n')
    f.close()
    return clfs


def base_gbdt_learners(group_num,local_time,hy1,hy2):
    clfs = []
    with open('./results/base learners parameter.txt','a') as f:
        f.write('------------- Base learners - '+ local_time +' ---------------\n')
        for i in range(group_num):
            print('GBGT'+str(i+1)+': n_estimators = '+str(hy1)+' learning_rate: '+str(hy2)+'\n')
            f.write('GBGT'+str(i+1)+'\tn_estimators\t'+str(hy1)+'\learning_rate\t'+str(hy2)+'\n')
            clfs.append(GradientBoostingClassifier(n_estimators=hy1, learning_rate=hy2))
        f.write('----------------------------------------------------------------------\n')
    f.close()
    return clfs


def soft_voting_strategy(base_preds, base_probs):
    print('\nSoft voting ...\n')
    pred_final = []
    prob_final = []
    for prob in base_probs:
        mean_prob = np.mean(prob)
        prob_final.append(mean_prob)
        if mean_prob > 0.5:
            pred_final.append(1)
        else:
            pred_final.append(0)
    return pred_final, prob_final