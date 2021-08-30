# -*- coding: utf-8 -*-
import import_dataset as dataset
import stacked_learning
import numpy as np
import test_scores as score
import time

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier


def ERMDA(data_path, fold_num, sel_fea, sel_fea_num, base_hp1, base_hp2, sel_hp1, sel_hp2, tolerance):
    count = 0
    best_auc = 0
    best_prec = 0
    best_recall = 0
    best_f1_score = 0
    best_auc = 0
    best_sum = 0
    best_base_learner_num = 0
    base_learner_num = 2
    while(count < tolerance and base_learner_num <= 20):
        local_time = time.asctime(time.localtime(time.time()))
        posi_data, unlabelled_data = dataset.import_dis_mir_vec(data_path)
        posi_num = len(posi_data)
        unlabelled_train_data, unlabelled_cv_data = dataset.spliting_unlabelled_data(unlabelled_data, posi_num)
        metric_dict = {}
        soft_ave_acc = 0
        soft_ave_prec = 0
        soft_ave_recall = 0
        soft_ave_f1_score = 0
        soft_ave_auc = 0
        soft_ave_sum = 0
        with open('./results/MDA GBDT process results '+str(sel_fea)+' sel.txt', 'a') as f:
            f.write('\n\n**************************************************************\n'+local_time+'\n')
            f.write('cross validation fold number\t'+ str(fold_num)+ '\tbase_learner_num\t'+ str(base_learner_num)+'\tfeature selection\t'+str(sel_fea_num)+'\n')
            for fold in range(fold_num):
                f.write('\n-------------------------Fold'+str(fold+1)+'---------------------------\n')
                posi_train_data = np.array([x for i, x in enumerate(posi_data) if i % fold_num != fold])
                posi_test_data = np.array([x for i, x in enumerate(posi_data) if i % fold_num == fold])
                nega_test_data = np.array([x for i, x in enumerate(unlabelled_cv_data) if i % fold_num == fold])
                X_test = np.concatenate((posi_test_data[:,:-2], nega_test_data[:,:-2]))
                y_test = np.concatenate((posi_test_data[:,-2], nega_test_data[:,-2])) #GBDT
                base_learners = stacked_learning.base_gbdt_learners(base_learner_num, local_time, base_hp1, base_hp2)
                metric_list = []
                trained_clfs = []
                most_imps_list = []
                for i in range(base_learner_num):
                    print('\nthe ', i+1, ' base learner training...')
                    base_train = dataset.generating_base_train_data(posi_train_data, unlabelled_train_data)
                    X_base_train = np.array(base_train[:,:-2])
                    y_base_train = np.array(base_train[:,-2]) # GBDT
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
                    y_prob = y_prob[:,1] # GBDT
                    tp, fp, tn, fn, acc, prec, recall, f1_score, auc = score.calculate_performace(len(y_pred), y_pred, y_prob, y_test) #GBDT
                    base_sum = acc+ prec+ recall+ f1_score+ auc
                    print('the ', i+1, ' base learner proformance: \n  Acc = \t', acc, '\n  prec = \t', prec, '\n  recall = \t', recall, '\n  f1_score = \t', f1_score, '\n  AUC = \t', auc, '\n  Sum = \t', base_sum)
                    f.write('the '+ str(i+1)+ ' base learner proformance: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(acc)+'\t  prec = \t'+ str(prec)+ '\t  recall = \t'+str(recall)+'\t  f1_score = \t'+str(f1_score)+'\t  AUC = \t'+ str(auc)+'\t  Sum = \t'+ str(base_sum)+'\n')
                    metric_list.append([acc, prec, recall, f1_score, auc, base_sum])
                metric_dict[fold] = metric_list
                
                if sel_fea == 'RF':
                    base_preds, base_probs = dataset.base_preds_probs_imp_by_rf(X_test, most_imps_list, trained_clfs)
                else:
                    base_preds, base_probs = dataset.base_preds_probs(X_test, trained_clfs)
                    
                pred_final, prob_final = stacked_learning.soft_voting_strategy(base_preds, base_probs)
                soft_tp, soft_fp, soft_tn, soft_fn, soft_acc, soft_prec, soft_recall, soft_f1_score, soft_auc = score.calculate_performace(len(pred_final), pred_final, prob_final, y_test) #GBDT
                soft_final_sum = soft_acc+ soft_prec+ soft_recall+ soft_f1_score+ soft_auc
                print('soft voting final proformance: \n  Acc = \t', soft_acc, '\n  prec = \t', soft_prec, '\n  recall = \t', soft_recall, '\n  f1_score = \t', soft_f1_score, '\n  AUC = \t', soft_auc, '\n  Sum = \t', soft_final_sum)
                f.write('soft voting final proformance: \ttp\t'+ str(soft_tp) + '\tfp\t'+ str(soft_fp) + '\ttn\t'+ str(soft_tn)+ '\tfn\t'+ str(soft_fn)+'\tAcc\t'+ str(soft_acc)+'\tprec\t'+ str(soft_prec)+ '\trecall\t'+str(soft_recall)+'\tf1_score\t'+str(soft_f1_score)+'\tAUC\t'+ str(soft_auc)+'\tSum\t'+ str(soft_final_sum)+'\n')
                soft_ave_acc += soft_acc
                soft_ave_prec += soft_prec
                soft_ave_recall += soft_recall
                soft_ave_f1_score += soft_f1_score
                soft_ave_auc += soft_auc
                soft_ave_sum += soft_final_sum
            
            print('\n-------------------------------------------------------')
            soft_ave_acc /= fold_num
            soft_ave_prec /= fold_num
            soft_ave_recall /= fold_num
            soft_ave_f1_score /= fold_num
            soft_ave_auc /= fold_num
            soft_ave_sum /= fold_num
            f.write('\n------------------- the final result of MDA - '+ local_time+' -------------------\n')
            score.base_learners_results(metric_dict, fold_num, base_learner_num, f)
            print('Soft voting final proformance: \n  Acc = ', soft_ave_acc, '\n  prec = ', soft_ave_prec, '\n  recall = ', soft_ave_recall, '\n  f1_score = ', soft_ave_f1_score, '\n  AUC = ', soft_ave_auc, '\n Sum = ', soft_ave_sum)
            f.write('Soft voting final proformance: \tFeature Selection\t'+str(sel_fea)+'\tfea_sel_num\t'+str(sel_fea_num)+'\tsel_hp1\t'+str(sel_hp1)+'\tsel_hp2\t'+str(sel_hp2)+'\tbase_learner_num\t'+str(base_learner_num)+'\tAcc\t'+str(soft_ave_acc)+ '\tprec\t'+str(soft_ave_prec)+ '\trecall\t'+str(soft_ave_recall)+ '\tf1_score\t'+ str(soft_ave_f1_score)+ '\tAUC\t'+ str(soft_ave_auc)+ '\tSum\t'+str(soft_ave_sum))
            
            if soft_ave_auc < best_auc:
                count += 1
            else:
                best_acc = soft_ave_acc
                best_prec = soft_ave_prec
                best_recall = soft_ave_recall
                best_f1_score = soft_ave_f1_score
                best_auc = soft_ave_auc
                best_sum = soft_ave_sum
                best_base_learner_num = base_learner_num
                count = 0
            base_learner_num +=1
    print('\n-------------------------------------------------------------------------')
    f_best = open('./results/GBDT best results '+str(sel_fea)+' sel.txt', 'a+')
    f_best.write('Best voting final proformance: \tbase_learner_num\t'+str(best_base_learner_num)+'\tn_est\t'+str(base_hp1)+'\tlr\t'+str(base_hp2)+'\tFeature Selection\t'+str(sel_fea)+'\tfea_sel_num\t'+str(sel_fea_num)+'\tsel_hp1\t'+str(sel_hp1)+'\tsel_hp2\t'+str(sel_hp2)+'\tAcc\t'+str(best_acc)+ '\tprec\t'+str(best_prec)+ '\trecall\t'+str(best_recall)+ '\tf1_score\t'+ str(best_f1_score)+ '\tAUC\t'+ str(best_auc)+ '\tSum\t'+str(best_sum)+'\n')
    f_best.close()
    print('best final proformance: \n  Acc = ', best_acc, '\n  prec = ', best_prec, '\n  recall = ', best_recall, '\n  f1_score = ', best_f1_score, '\n  AUC = ', best_auc, '\n Sum = ', best_sum)
    f.close()
    
if __name__ == '__main__':
    data_path = './data/'
    fold_num = 5
    sel_fea = 'RF'
    sel_hp1 = 300
    sel_hp2 = 20
    tolerance = 5
    base_hp1 = 500
    base_hp2 = 0.4
    prop=0.4
    sel_fea_num = int(prop*878)
    ERMDA(data_path, fold_num, sel_fea, sel_fea_num, base_hp1, base_hp2, sel_hp1, sel_hp2, tolerance)