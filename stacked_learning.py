# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import numpy as np

def base_xgb_learners(group_num,local_time,hy1,hy2,hy3):
    clfs = []
    for i in range(group_num):
        print('XGB'+str(i+1)+': n_estimators = '+str(hy1)+' learning_rate: '+str(hy2)+'\n')
        clfs.append(XGBClassifier(n_estimators=hy1, max_depth=hy2, learning_rate=hy3))
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