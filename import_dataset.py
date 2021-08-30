# -*- coding: utf-8 -*-
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
#from keras.tensorflow.models import Sequential
#from keras.tensorflow.layers import Dense, Dropout
import xlrd


def import_dis_mir_vec(data_path):
    print('\nDisease comprehensive similarity ...\n')
    DSSM1 = np.loadtxt(data_path + 'D_SSM1.txt')
    DSSM2 = np.loadtxt(data_path + 'D_SSM2.txt')
    DSSM = (DSSM1 + DSSM2) / 2
    DGSM = np.loadtxt(data_path + 'D_GSM.txt')
    ID = np.zeros(shape = (DSSM.shape[0], DSSM.shape[1]))
    for i in range(DSSM.shape[0]):
        for j in range(DSSM.shape[1]):
            if DSSM[i][j] == 0:
                ID[i][j] = DGSM[i][j]
            else:
                ID[i][j] = DSSM[i][j]
    
    print('\nmiRNA comprehensive similarity ...\n')
    MFSM = np.loadtxt(data_path + 'M_FSM.txt')
    MGSM = np.loadtxt(data_path + 'M_GSM.txt')
    IM = np.zeros(shape = (MFSM.shape[0], MFSM.shape[1]))
    for i in range(MFSM.shape[0]):
        for j in range(MFSM.shape[1]):
            if MFSM[i][j] == 0:
                IM[i][j] = MGSM[i][j]
            else:
                IM[i][j] = MFSM[i][j]
                
    print('\nconstruct positive pairs and unlabelled pairs ...\n')
    A = np.zeros(shape = (DSSM.shape[0], MFSM.shape[1]))
    asso_file =  xlrd.open_workbook(data_path + 'Known disease-miRNA association number.xlsx')
    asso_pairs = asso_file.sheets()[0]
    for i in range(5430):
        asso = asso_pairs.row_values(i)
        m = int(asso[0])
        n = int(asso[1])
        A[n-1, m-1] = 1
        
    known=[]
    unknown=[]                         
    for x in range(383):
        for y in range(495):
            if A[x,y]==0:                 
                unknown.append((x,y))
            else:
                known.append((x,y))
    
    posi_list = []
    for i in range(5430):
        posi = ID[known[i][0],:].tolist() + IM[known[i][1],:].tolist() + [1, 0]
        posi_list.append(posi)
    unlabelled_list = []
    for j in range(184155):
        unlabelled=ID[unknown[j][0],:].tolist()+IM[unknown[j][1],:].tolist() + [0, 1]
        unlabelled_list.append(unlabelled)
    
    random.shuffle(posi_list)
    random.shuffle(unlabelled_list)
    return posi_list, unlabelled_list


# Divide the same number of unlabeled samples as the positive samples
def spliting_unlabelled_data(unlabelled_data, test_num):
    unlabelled_train_data = unlabelled_data[test_num:]
    unlabelled_cv_data = unlabelled_data[:test_num]
    return unlabelled_train_data, unlabelled_cv_data


def generating_base_train_data(posi_train_data, unlabelled_train_data):
    sample_num = posi_train_data.shape[0]
    samples = random.sample(unlabelled_train_data, sample_num)
    #print('group',i+1,'   ',len(samples+posi_train_data))
    base_train = np.concatenate((posi_train_data, samples))
    return base_train


# Feature importance ranking for each base training dataset
def feature_ranking_by_rf(data, label, fea_num,sel_hp1,sel_hp2):
    print('RF Feature selection (', fea_num, ') ...\n')
    fs_rf = RandomForestClassifier(n_estimators=sel_hp1, max_depth=sel_hp2, random_state=0)
    fs_rf.fit(data, label)
    importances = fs_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    most_imp = indices[:fea_num]
    #data = data[:,most_imp]
    return most_imp


# non-voting strategy -- generate new dataset
def generating_new_data(X_train, X_test, trained_clfs):
    X_train_prob_list = []
    X_test_prob_list = []
    for clf in trained_clfs:
        X_train_prob = clf.predict_proba(X_train)
        X_train_prob_list.append(X_train_prob[1][:,0])
        X_test_prob = clf.predict_proba(X_test)
        X_test_prob_list.append(X_test_prob[1][:,0])
    #print(len(posi_train_prob_list[0]),'    ',len(unlabelled_train_prob_list[0]))
    X_train_prob_list = np.array(X_train_prob_list)
    X_test_prob_list = np.array(X_test_prob_list)
    X_train_new_data = []
    for i in range(len(X_train_prob_list[0])):
        X_train_new_data.append(X_train_prob_list[:,i])
    #posi_train_new_data = np.array(posi_train_new_data)
    X_test_new_data = []
    for i in range(len(X_test_prob_list[0])):
        X_test_new_data.append(X_test_prob_list[:,i])
    #posi_test_new_data = np.array(posi_test_new_data)
    X_train_new_data = np.array(X_train_new_data)
    X_test_new_data = np.array(X_test_new_data)
    return X_train_new_data, X_test_new_data


# non-voting strategy -- generate new dataset & feature selection
def generating_new_data_imp(X_train, X_test, most_imps_list, trained_clfs):
    X_train_prob_list = []
    X_test_prob_list = []
    count = 0
    for clf in trained_clfs:
        most_imps = most_imps_list[count]
        #print(most_imps)
        X_train_si = X_train[:,most_imps]
        X_test_si = X_test[:,most_imps]
        X_train_prob = clf.predict_proba(X_train_si)
        X_train_prob_list.append(X_train_prob[1][:,0])
        X_test_prob = clf.predict_proba(X_test_si)
        X_test_prob_list.append(X_test_prob[1][:,0])
        count = count + 1
    #print(len(posi_train_prob_list[0]),'    ',len(unlabelled_train_prob_list[0]))
    X_train_prob_list = np.array(X_train_prob_list)
    X_test_prob_list = np.array(X_test_prob_list)
    X_train_new_data = []
    for i in range(len(X_train_prob_list[0])):
        X_train_new_data.append(X_train_prob_list[:,i])
    #posi_train_new_data = np.array(posi_train_new_data)
    X_test_new_data = []
    for i in range(len(X_test_prob_list[0])):
        X_test_new_data.append(X_test_prob_list[:,i])
    #posi_test_new_data = np.array(posi_test_new_data)
    X_train_new_data = np.array(X_train_new_data)
    X_test_new_data = np.array(X_test_new_data)
    return X_train_new_data, X_test_new_data


# voting strategy, return preds and probs of base learners
def base_preds_probs(X_test, trained_clfs):
    prob_list = []
    pred_list = []
    X_test = np.array(X_test)
    count = 0
    for clf in trained_clfs:
        pred = clf.predict(X_test)
        pred_list.append(pred)
        prob = clf.predict_proba(X_test)
        prob_list.append(prob[:,1])
        count = count + 1
    pred_list = np.array(pred_list)
    prob_list = np.array(prob_list)
    base_preds = []
    base_probs = []
    for i in range(len(pred_list[0])):
        base_preds.append(pred_list[:,i])
    for i in range(len(prob_list[0])):
        base_probs.append(prob_list[:,i])
    
    return base_preds, base_probs


# voting strategy & feature selection, return preds and probs of base learners
def base_preds_probs_imp_by_rf(X_test, most_imps_list, trained_clfs):
    prob_list = []
    pred_list = []
    X_test = np.array(X_test)
    count = 0
    for clf in trained_clfs:
        most_imp = most_imps_list[count]
        X_test_si = X_test[:,most_imp]
        pred = clf.predict(X_test_si)
        pred_list.append(pred)
        prob = clf.predict_proba(X_test_si)
        prob_list.append(prob[:,1])
        count = count + 1
    pred_list = np.array(pred_list)
    prob_list = np.array(prob_list)
    base_preds = []
    base_probs = []
    for i in range(len(pred_list[0])):
        base_preds.append(pred_list[:,i])
    for i in range(len(prob_list[0])):
        base_probs.append(prob_list[:,i])
    
    return base_preds, base_probs