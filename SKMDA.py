# -*- coding: utf-8 -*-
import import_dataset as dataset
import stacked_learning
import numpy as np
import test_scores as score
import time

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier


def ERMDA(data_path, fold_num, sel_fea, sel_fea_num, base_hp1, base_hp2, base_hp3, sel_hp1, sel_hp2):
    base_learner_num = 13
    local_time = time.asctime(time.localtime(time.time()))
    posi_data, unlabelled_data = dataset.import_dis_mir_vec(data_path)
    posi_num = len(posi_data)
    unlabelled_train_data, unlabelled_cv_data = dataset.spliting_unlabelled_data(unlabelled_data, posi_num)
    metric_dict = {}
    soft_acc_list = []
    soft_prec_list = []
    soft_recall_list = []
    soft_f1_score_list = []
    soft_auc_list = []
    soft_aupr_list = []
    with open('./results/ERMDA process results '+str(sel_fea)+' sel.txt', 'a') as f:
        f.write('\n\n**************************************************************\n'+local_time+'\n')
        f.write('cross validation fold number\t'+ str(fold_num)+ '\tbase_learner_num\t'+ str(base_learner_num)+'\tfeature selection\t'+str(sel_fea_num)+'\n')
        for fold in range(fold_num):
            f.write('\n-------------------------Fold'+str(fold+1)+'---------------------------\n')
            posi_train_data = np.array([x for i, x in enumerate(posi_data) if i % fold_num != fold])
            posi_test_data = np.array([x for i, x in enumerate(posi_data) if i % fold_num == fold])
            nega_test_data = np.array([x for i, x in enumerate(unlabelled_cv_data) if i % fold_num == fold])
            X_test = np.concatenate((posi_test_data[:,:-2], nega_test_data[:,:-2]))
            y_test = np.concatenate((posi_test_data[:,-2], nega_test_data[:,-2]))
            base_learners = stacked_learning.base_rf_learners(base_learner_num, local_time, base_hp1, base_hp2)
            metric_list = []
            trained_clfs = []
            most_imps_list = []
            for i in range(base_learner_num):
                print('\nthe ', i+1, ' base learner training...')
                base_train = dataset.generating_base_train_data(posi_train_data, unlabelled_train_data)
                X_base_train = np.array(base_train[:,:-2])
                y_base_train = np.array(base_train[:,-2])
                X_base_test = X_test
                if sel_fea == 'RF':
                    print('\nRF feature ranking ...\n')
                    most_imps = dataset.feature_ranking_by_rf(X_base_train, y_base_train, sel_fea_num,sel_hp1,sel_hp2)
                    most_imps_list.append(most_imps)
                    X_base_train = X_base_train[:,most_imps]
                    X_base_test = X_test[:,most_imps]
                    
                print('the ', i+1, 'training group X shape: ', X_base_train.shape)
                print('the ', i+1, 'training group y shape: ', y_base_train.shape)
                clf = base_learners[i]
                clf.fit(X_base_train,y_base_train)
                trained_clfs.append(clf)
                
                # evaluate the i base learner performance
                print('the ', i+1, 'testing group X shape: ', X_base_test.shape)
                print('the ', i+1, 'testing group y shape: ', y_test.shape)
                y_pred = clf.predict(X_base_test)
                y_prob = clf.predict_proba(X_base_test)
                y_prob = y_prob[:,1]
                tp, fp, tn, fn, acc, prec, recall, f1_score, auc,aupr = score.calculate_performace(len(y_pred), y_pred, y_prob, y_test) 
                print('the ', i+1, ' base learner proformance: \n  Acc = \t', acc, '\n  prec = \t', prec, '\n  recall = \t', recall, '\n  f1_score = \t', f1_score, '\n  AUC = \t', auc, '\n  aupr = \t', aupr)
                f.write('the '+ str(i+1)+ ' base learner proformance: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(acc)+'\t  prec = \t'+ str(prec)+ '\t  recall = \t'+str(recall)+'\t  f1_score = \t'+str(f1_score)+'\t  AUC = \t'+ str(auc)+'\t  AUPR = \t'+ str(aupr)+'\n')
                metric_list.append([acc, prec, recall, f1_score, auc, aupr])
            metric_dict[fold] = metric_list
            
            if sel_fea == 'RF':
                base_preds, base_probs = dataset.base_preds_probs_imp_by_rf(X_test, most_imps_list, trained_clfs)
            else:
                base_preds, base_probs = dataset.base_preds_probs(X_test, trained_clfs)
                
            pred_final, prob_final = stacked_learning.soft_voting_strategy(base_preds, base_probs)
            soft_tp, soft_fp, soft_tn, soft_fn, soft_acc, soft_prec, soft_recall, soft_f1_score, soft_auc, soft_aupr = score.calculate_performace(len(pred_final), pred_final, prob_final, y_test)
            print('XGB_RF_sel soft voting proformance: \n  Acc = \t', soft_acc, '\n  prec = \t', soft_prec, '\n  recall = \t', soft_recall, '\n  f1_score = \t', soft_f1_score, '\n  AUC = \t', soft_auc, '\n  AUPR = \t', soft_aupr)
            f.write('XGB_RF_sel soft voting proformance: \ttp\t'+ str(soft_tp) + '\tfp\t'+ str(soft_fp) + '\ttn\t'+ str(soft_tn)+ '\tfn\t'+ str(soft_fn)+'\tAcc\t'+ str(soft_acc)+'\tprec\t'+ str(soft_prec)+ '\trecall\t'+str(soft_recall)+'\tf1_score\t'+str(soft_f1_score)+'\tAUC\t'+ str(soft_auc)+'\tAUPR\t'+ str(soft_aupr)+'\n')
            soft_acc_list.append(soft_acc)
            soft_prec_list.append(soft_prec)
            soft_recall_list.append(soft_recall)
            soft_f1_score_list.append(soft_f1_score)
            soft_auc_list.append(soft_auc)
            soft_aupr_list.append(soft_aupr)
        
        print('\n-------------------------------------------------------')
        soft_acc_arr = np.array(soft_acc_list)
        soft_prec_arr = np.array(soft_prec_list)
        soft_recall_arr = np.array(soft_recall_list)
        soft_f1_score_arr = np.array(soft_f1_score_list)
        soft_auc_arr = np.array(soft_auc_list)
        soft_aupr_arr = np.array(soft_aupr_list)
        
        soft_ave_acc = np.mean(soft_acc_arr)
        soft_ave_prec = np.mean(soft_prec_arr)
        soft_ave_recall = np.mean(soft_recall_arr)
        soft_ave_f1_score = np.mean(soft_f1_score_arr)
        soft_ave_auc = np.mean(soft_auc_arr)
        soft_ave_aupr = np.mean(soft_aupr_arr)

        soft_std_acc = np.std(soft_acc_arr)
        soft_std_prec = np.std(soft_prec_arr)
        soft_std_recall = np.std(soft_recall_arr)
        soft_std_f1_score = np.std(soft_f1_score_arr)
        soft_std_auc = np.std(soft_auc_arr)
        soft_std_aupr = np.std(soft_aupr_arr)
        f.write('\n------------------- the final result of MDA - '+ local_time+' -------------------\n')
        score.base_learners_results(metric_dict, fold_num, base_learner_num, f)
        print('XGB_RF_sel Final proformance: \n  Acc = ', soft_ave_acc, '\n  prec = ', soft_ave_prec, '\n  recall = ', soft_ave_recall, '\n  f1_score = ', soft_ave_f1_score, '\n  AUC = ', soft_ave_auc, '\n AUPR = ', soft_aupr)
        f.write('XGB_RF_sel Final proformance: \tFeature Selection\t'+str(sel_fea)+'\tfea_sel_num\t'+str(sel_fea_num)+'\tbase_hp1\t'+str(base_hp1)+'\tbase_hp2\t'+str(base_hp2)+'\tbase_hp3\t'+str(base_hp3)+'\tbase_learner_num\t'+str(base_learner_num)+'\tAcc\t'+str(soft_ave_acc)+'&'+str(soft_std_acc)+ '\tprec\t'+str(soft_ave_prec)+'&'+str(soft_std_prec)+ '\trecall\t'+str(soft_ave_recall)+'&'+str(soft_std_recall)+ '\tf1_score\t'+ str(soft_ave_f1_score)+'&'+str(soft_std_f1_score)+ '\tAUC\t'+ str(soft_ave_auc)+'&'+str(soft_std_auc)+ '\tAUPR\t'+str(soft_ave_aupr)+'&'+str(soft_std_aupr))
    f.close()
    
if __name__ == '__main__':
    data_path = './data/'
    fold_num = 5
    tolerance = 3
    base_hp1 = 500
    base_hp2 = 5
    base_hp3 = 0.4
    
    sel_fea = 'RF'
    prop = 0.5
    sel_hp1 = 300
    sel_hp2 = 30
    sel_fea_num = int(prop*878)
    ERMDA(data_path, fold_num, sel_fea, sel_fea_num, base_hp1, base_hp2, base_hp3, sel_hp1, sel_hp2)
