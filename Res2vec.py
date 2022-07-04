# -*- coding: utf-8 -*-

"""
测试size-windows,酒酿酵母数据集
"""

import os
from time import time
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Conv1D, Reshape, MaxPooling1D, ZeroPadding1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import roc_curve, auc, roc_auc_score,average_precision_score
import numpy as np
from keras.layers import Dense, Dropout, Merge
import utils.tools as utils
from keras.regularizers import l2
from gensim.models import Word2Vec
import copy
import psutil
import random
import h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.initializers import glorot_uniform
from keras.layers import  Add, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from help import identify_block,convolutional_block,token,pandding_J,read_traingingData,protein_representation,mkdir,getMemorystate
from keras.layers.merge import concatenate
from keras.models import Model
from attention import *
lr = 0.01
epochs = 5
batch_size = 64
dr = 0.4
l2c = 0.01

def define_model(sequence_len):
    
    ########################################################"Channel-1" ########################################################
    m=sequence_len

    input_1 = Input(shape=(m,), name='Protein_a')

    p11 = Dense(1024, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_111',
                kernel_regularizer=l2(l2c))(input_1)
    p11 = BatchNormalization(axis=-1)(p11)
    p11 = Dropout(dr)(p11)

    p12 = Dense(512, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_1',
                kernel_regularizer=l2(l2c))(p11)
    p12 = BatchNormalization(axis=-1)(p12)
    p12 = Dropout(dr)(p12)

    p13 = Dense(256, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_2',
                kernel_regularizer=l2(l2c))(p12)
    p13 = BatchNormalization(axis=-1)(p13)
    p13 = Dropout(dr)(p13)

    p14 = Dense(128, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_3',
                kernel_regularizer=l2(l2c))(p13)
    p14 = BatchNormalization(axis=-1)(p14)
    p14 = Dropout(dr)(p14)


    ########################################################"Channel-2" ########################################################

    input_2 = Input(shape=(m,), name='Protein_b')

    p21 = Dense(1024, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_111',
                kernel_regularizer=l2(l2c))(input_2)
    p21 = BatchNormalization(axis=-1)(p21)
    p21 = Dropout(dr)(p21)

    p22 = Dense(512, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_1',
                kernel_regularizer=l2(l2c))(p21)
    p22 = BatchNormalization(axis=-1)(p22)
    p22 = Dropout(dr)(p22)

    p23 = Dense(256, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_2',
                kernel_regularizer=l2(l2c))(p22)
    p23 = BatchNormalization(axis=-1)(p23)
    p23 = Dropout(dr)(p23)

    p24 = Dense(128, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_3',
                kernel_regularizer=l2(l2c))(p23)
    p24 = BatchNormalization(axis=-1)(p24)
    p24 = Dropout(dr)(p24)

    ##################################### Merge Abstraction features ##################################################

    merged = concatenate([p14, p24], name='merged_protein1_2')

    ##################################### Prediction Module ##########################################################

    pre_output = Dense(64, activation='relu', kernel_initializer='glorot_normal', name='Merged_feature_2')(merged)
    pre_output = Dense(32, activation='relu', kernel_initializer='he_uniform', name='Merged_feature_3')(pre_output)
    pre_output = Dense(8, activation='relu', kernel_initializer='he_uniform', name='Merged_feature_4')(pre_output)

    pre_output = Dropout(dr)(pre_output)

    output = Dense(1, activation='sigmoid', name='output')(pre_output)

    model = Model(inputs=[input_1, input_2], output=output)

    sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


              
def get_training_dataset(wv,  maxlen,size):

    file_1 = '../S.score/S.score P1.txt'
    file_2 = '../S.score/S.score P2.txt'
    file_3 = '../S.score/S.score N1.txt'
    file_4 = '../S.score/S.score N2.txt'
    # positive seq protein A
    pos_seq_protein_A = read_traingingData(file_1)
    pos_seq_protein_B = read_traingingData(file_2)
    neg_seq_protein_A = read_traingingData(file_3)
    neg_seq_protein_B = read_traingingData(file_4)
    # put pos and neg together
    pos_neg_seq_protein_A = copy.deepcopy(pos_seq_protein_A)   
    pos_neg_seq_protein_A.extend(neg_seq_protein_A)
    pos_neg_seq_protein_B = copy.deepcopy(pos_seq_protein_B)   
    pos_neg_seq_protein_B.extend(neg_seq_protein_B)
    # token
    token_pos_neg_seq_protein_A = token(pos_neg_seq_protein_A)
    token_pos_neg_seq_protein_B = token(pos_neg_seq_protein_B)
    # padding
    tokened_token_pos_neg_seq_protein_A = pandding_J(token_pos_neg_seq_protein_A, maxlen)
    tokened_token_pos_neg_seq_protein_B = pandding_J(token_pos_neg_seq_protein_B,maxlen)
    # protein reprsentation
    feature_protein_A  = protein_representation(wv,tokened_token_pos_neg_seq_protein_A,maxlen,size)
    feature_protein_B  = protein_representation(wv,tokened_token_pos_neg_seq_protein_B,maxlen,size)
    feature_protein_AB = np.hstack((np.array(feature_protein_A),np.array(feature_protein_B)))
    mem_hstack = getMemorystate()
    #  creat label
    label = np.ones(len(feature_protein_A))
    label[len(feature_protein_AB)//2:] = 0
   
    return feature_protein_AB,label
                         
def res2vec(sizes,windows,maxlens):

    for size in sizes:
        for window in windows:
            sg = 'wv_wv_swissProt_size_'+str(size)+'_window_'+str(window) 
            model_wv = Word2Vec.load('/word2vec/'+sg+'.model')
            scores=[]            
            for maxlen in maxlens:
                
                # get training data  
                train_fea_protein_AB, label= get_training_dataset(model_wv.wv, maxlen,size )

                scaler = StandardScaler().fit(train_fea_protein_AB)
                train_fea_protein_AB = scaler.transform(train_fea_protein_AB)
                X = np.array(train_fea_protein_AB)
                sequence_len = size*maxlen
                X1_train = X[:,:sequence_len] 
                X2_train = X[:,sequence_len:]
                y=label
                kf = StratifiedKFold(n_splits=5)
                o=0
                k=1

                for train, test in kf.split(X,y):
                    
                    global model
                    model = define_model(sequence_len)
                    o = o + 1
                    k = k + 1
                    model.fit([X1_train[train], X2_train[train]], y[train], epochs=50, batch_size=batch_size, verbose=1)

                    y_test = y[test]

                    y_score = model.predict([X1_train[test], X2_train[test]])       
        
                    auc_test = roc_auc_score(y_test, y_score)
                    pr_test = average_precision_score(y_test, y_score)

                    for i in range(0,len(y_score)):
                        if(y_score[i]>0.5):
                            y_score[i]=1
                        else:
                            y_score[i]=0

                    tp_test,fp_test,tn_test,fn_test,accuracy_test, precision_test, sensitivity_test,recall_test, specificity_test, MCC_test, f1_score_test,_,_,_= utils.calculate_performace(len(y_score), y_score, y_test)

                    print('\ttp=%0.0f,fp=%0.0f,tn=%0.0f,fn=%0.0f'%(tp_test,fp_test,tn_test,fn_test))
                    print('\tacc=%0.4f,pre=%0.4f,rec=%0.4f,sp=%0.4f,mcc=%0.4f,f1=%0.4f'
                % (accuracy_test, precision_test, recall_test, specificity_test, MCC_test, f1_score_test))
                    print('\tauc=%0.4f,pr=%0.4f'%(auc_test,pr_test))
                    scores.append([accuracy_test,precision_test, recall_test,specificity_test, MCC_test, f1_score_test, auc_test,pr_test]) 
                
                            ##########################创建保存文件##############################
                import sys  # 需要引入的包


                # 以下为包装好的 Logger 类的定义
                class Logger(object):
                    def __init__(self, filename="Default.log"):
                        self.terminal = sys.stdout
                        # self.log = open(filename, "a")
                        self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误

                    def write(self, message):
                        self.terminal.write(message)
                        self.log.write(message)

                    def flush(self):
                        pass
                #########################保存以下信息###############################
                sys.stdout = Logger('/encoding.txt')
                # sc= pd.DataFrame(scores)   
                # sc.to_csv(result_dir+'5cv_'+db+'_scores.csv')   
                scores_array = np.array(scores)
                # print (db+'_5cv:')
                print('wv_swissProt_size_'+str(size)+'_window_'+str(window) )
                print(("accuracy=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[0]*100,np.std(scores_array, axis=0)[0]*100)))
                print(("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[1]*100,np.std(scores_array, axis=0)[1]*100)))
                print("recall=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[2]*100,np.std(scores_array, axis=0)[2]*100))
                print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[3]*100,np.std(scores_array, axis=0)[3]*100))
                print("MCC=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[4]*100,np.std(scores_array, axis=0)[4]*100))
                print("f1_score=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[5]*100,np.std(scores_array, axis=0)[5]*100))
                print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[6]*100,np.std(scores_array, axis=0)[6]*100))
                print("roc_pr=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[7]*100,np.std(scores_array, axis=0)[7]*100))
                                                             
#%%  
if __name__ == "__main__":  
    sizes = [ 2,4,6,8,10,12,14,16,18,20] 
    windows = [4,8,12,16,20,22]
    maxlens = [810]
    print('**************************************')
    print('**************************************')
    print('res2vec:')
    train_fea_protein_AB = res2vec(sizes,windows,maxlens) 
    print('**************************************')   
    print('**************************************')
     
        
                                                      
                                    
                                
                
                
                
                
                
                
