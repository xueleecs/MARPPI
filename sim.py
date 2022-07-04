# from typing import Concatenate
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, Dropout, Conv1D, Reshape, MaxPooling1D, ZeroPadding1D, AveragePooling1D, Lambda
from keras.layers import  Add, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.models import Model
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from keras.utils import np_utils
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef,accuracy_score, precision_recall_curve, precision_score,recall_score,f1_score,cohen_kappa_score,auc
from sklearn.manifold import TSNE
from keras.layers import Bidirectional, LSTM, Lambda, K, RepeatVector, Permute, merge,GRU
from keras.initializers import glorot_uniform
from gensim.models import Word2Vec
import time
import tensorflow as tf
from sklearn.metrics import matthews_corrcoef,accuracy_score, precision_score,recall_score
import copy
from help import token,pandding_J,read_traingingData,protein_representation,mkdir,getMemorystate


start = time.time()
dr=0.2
dr3=0.25
lr=0.003

def get_training_dataset(wv,  maxlen,size):

    file_1 = '/S.score/S.score P1.txt'
    file_2 = '/S.score/S.score P2.txt'
    file_3 = '/S.score/S.score N1.txt'
    file_4 = '/S.score/S.score N2.txt'
    # positive seq protein A
    pos_seq_protein_A = read_traingingData(file_1)#5594
    pos_seq_protein_B = read_traingingData(file_2)
    neg_seq_protein_A = read_traingingData(file_3)
    neg_seq_protein_B = read_traingingData(file_4)
    # put pos and neg together
    pos_neg_seq_protein_A = copy.deepcopy(pos_seq_protein_A)   
    pos_neg_seq_protein_A.extend(neg_seq_protein_A)#11188
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
    label = np.ones(len(feature_protein_A))
    label[len(feature_protein_AB)//2:] = 0
   
    return feature_protein_AB,label

def res2vec(sizes,windows,maxlens):

    for size in sizes:
        for window in windows:
            sg = 'wv_wv_swissProt_size_'+str(size)+'_window_'+str(window) 
            model_wv = Word2Vec.load('/word2vec/'+sg+'.model')
           
            for maxlen in maxlens:
                
                # get training data  
                train_fea_protein_AB, label= get_training_dataset(model_wv.wv, maxlen,size )
                #11188 6480
                #scaler
                scaler = StandardScaler().fit(train_fea_protein_AB)
                train_fea_protein_AB = scaler.transform(train_fea_protein_AB)
                X = np.array(train_fea_protein_AB)
                return X   

def identify_block(X, f, filters, stage, block):
    """
    X - 输入的tensor类型数据，维度为（m, n_H_prev, n_W_prev, n_H_prev）
    f - kernal大小
    filters - 整数列表，定义每一层卷积层过滤器的数量
    stage - 整数 定义层位置
    block - 字符串 定义层位置

    X - 恒等输出，tensor类型，维度（n_H, n_W, n_C）
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters  # 定义输出特征的个数
    X_shortcut = X

    X = Conv1D(filters=F1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X)
    # 没有激活

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    # 参数意义和上文相同
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X 

    X = Conv1D(filters=F1, kernel_size=1, strides=s, padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2c')(X) 

    # shortcut
    X_shortcut = Conv1D(filters=F3, kernel_size=1, strides=s, padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(X_shortcut) 

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def identify_blocknew(X, f, filters, stage, block, s=2):
    # 参数意义和上文相同
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X   #(?, 2, 256)

    X = Conv1D(filters=F1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    

    M= Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 4})(X)
    M0=M[0]
    M1=M[1]
    M2=M[2]
    M3=M[3]
    
    l=M0.shape.as_list()[2]
    N0=M0
    N1=Conv1D(filters=l, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b1',
               kernel_initializer=glorot_uniform(seed=0))(M1)
    M2=Add()([M2,N1])
    N2=Conv1D(filters=l, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b2',
            kernel_initializer=glorot_uniform(seed=0))(M2)
    M3=Add()([M3,N2])
    N3=Conv1D(filters=l, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b3',
            kernel_initializer=glorot_uniform(seed=0))(M3)
    X=concatenate([N0,N1,N2,N3])

    X = Conv1D(filters=F3, kernel_size=1, strides=1, padding='same', name=conv_name_base + '2b4',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # shortcut
    X_shortcut = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(X_shortcut)  #(?, 1, 256)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def define_model():
    ########################################################"Channel-1" ########################################################
    
    input_1 = Input(shape=(16539, ), name='Protein_a')

    p1 = Reshape((111, 149))(input_1)

    #stage1
    p1 = Conv1D(64, kernel_size=7,strides=2, name='conv1')(p1) 
    p1 = BatchNormalization(axis=-1)(p1)
    X = Activation('relu')(p1) 
    X = MaxPooling1D(pool_size=3, strides=2)(X)


    #stage2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=21, block='a', s=1) 
    X = identify_block(X, f=3, filters=[64, 64, 256], stage=21, block='b')
    X = identify_blocknew(X, f=3, filters=[128, 128, 512], stage=21, block='c')

    # 均值池化
    p1 = AveragePooling1D(pool_size=2, padding='same')(X) #shape=(?, 1, 2048)

    ########################################################"Channel-2" ########################################################

    input_2 = Input(shape=(16539, ), name='Protein_b')

    p2 = Reshape((111, 149))(input_2)

    #stage1
    p2 = Conv1D(64, kernel_size=7,strides=2,name='conv')(p2)
    p2 = BatchNormalization(axis=-1)(p2)
    p2 = Activation('relu')(p2)
    p2 = MaxPooling1D(pool_size=3, strides=2)(p2)

    #stage2
    p2 = convolutional_block(p2, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    p2 = identify_block(p2, f=3, filters=[64, 64, 256], stage=2, block='b')
    p2 = identify_blocknew(p2, f=3, filters=[128, 128, 512], stage=2, block='c')

    # 均值池化
    p2 = AveragePooling1D(pool_size=2, padding='same')(p2)
    #################################### Merge Abstraction features ##################################################
    
    merged = concatenate([p1,p2], name='merged_protein1_2')
    ##################################### Prediction Module ##########################################################
    merged = Flatten()(merged)

    pre_output = Dense(512, activation='relu', kernel_initializer='glorot_normal', name='Merged_feature_1')(merged)
    pre_output = BatchNormalization()(pre_output)
    pre_output = Dropout(dr)(pre_output)
 
    pre_output = Dense(128, activation='relu', kernel_initializer='glorot_normal', name='Merged_feature_3')(merged)
    pre_output = BatchNormalization()(pre_output)
    pre_output = Dropout(dr)(pre_output)
   
    pre_output = Dense(32, activation='relu', kernel_initializer='he_uniform', name='Merged_feature_5')(pre_output)
    pre_output = BatchNormalization()(pre_output)
    pre_output = Dropout(dr)(pre_output)
 
    pre_output = Dense(8, activation='relu', kernel_initializer='he_uniform', name='Merged_feature_7')(pre_output)
    pre_output = BatchNormalization()(pre_output)

    pre_output=Dropout(dr)(pre_output)

    output = Dense(1, activation='sigmoid', name='output')(pre_output)

    model = Model(input=[input_1,input_2], output=output)

    sgd = SGD(lr=lr, momentum=0.9, decay=lr/k)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


##################################### Load Positive and Negative Dataset ##########################################################

##load resnet data1    
sizes= [20]
windows = [12]
maxlens= [810]
train_fea_protein_AB = res2vec(sizes,windows,maxlens)  

m=5594  #S.score

df_posR=train_fea_protein_AB[0:m,:]
df_negR=train_fea_protein_AB[m:,:]
df_posR1=df_posR[:,:16200]
df_posR2=df_posR[:,16200:]
df_negR1=df_negR[:,:16200]
df_negR2=df_negR[:,16200:]

df_posG = pd.read_csv('/S.score/NO.2/Positive.csv',header=None)
df_negG = pd.read_csv('/S.score/NO.2/Negative.csv',header=None)

df_posG = df_posG.iloc[:,0:678].values
scaler = StandardScaler().fit(df_posG)
df_posG = scaler.transform(df_posG)

df_negG = df_negG.iloc[:,0:678].values
scaler = StandardScaler().fit(df_negG)
df_negG = scaler.transform(df_negG)

df_posG1=df_posG[:,:339]
df_posG2=df_posG[:,339:]
df_negG1=df_negG[:,:339]
df_negG2=df_negG[:,339:]

df_posR=pd.DataFrame(df_posR)
df_negR=pd.DataFrame(df_negR)
df_posG=pd.DataFrame(df_posG)
df_negG=pd.DataFrame(df_negG)

df_posR1=pd.DataFrame(df_posR1)
df_negR1=pd.DataFrame(df_negR1)
df_posG1=pd.DataFrame(df_posG1)
df_negG1=pd.DataFrame(df_negG1)
df_posR2=pd.DataFrame(df_posR2)
df_negR2=pd.DataFrame(df_negR2)
df_posG2=pd.DataFrame(df_posG2)
df_negG2=pd.DataFrame(df_negG2)
##concat

df_pos = pd.concat([df_posR1,df_posG1,df_posR2,df_posG2],axis=1)
df_neg = pd.concat([df_negR1,df_negG1,df_negR2,df_negG2],axis=1)

df_pos=df_pos[~np.isnan(df_pos).any(axis=1)]
df_neg=df_neg[~np.isnan(df_neg).any(axis=1)]

df_neg['Status'] = 0
df_pos['Status'] = 1
df_neg=df_neg.sample(n=len(df_pos))

#取正、负样本的比例
df_pos=df_pos.sample(frac=1)
df_neg=df_neg.sample(frac=1)
df = pd.concat([df_pos,df_neg])

X = df.iloc[:,0:33078].values
y = df.iloc[:,33078:].values
# Trainlabels=y

X1_train = X[:, :16539]
X2_train = X[:, 16539:]
##################################### Five-fold Cross-Validation ##########################################################
    
kf=StratifiedKFold(n_splits=5)

accuracy1 = []
specificity1 = []
sensitivity1 = []
precision1=[]
recall1=[]
F11=[]
kappa1=[]

m_coef=[]
dnn_fpr_list=[]
dnn_tpr_list=[]
dnn_auc_list = []
aupr_list=[]
o=0
k=1
max_accuracy=float("-inf")
dnn_fpr=None
dnn_tpr=None

for train, test in kf.split(X,y):
    global model
    model=define_model()
    o=o+1
    k=k+1
    model.fit([X1_train[train],X2_train[train]],y[train],epochs=50,batch_size=64,verbose=0)

    y_test=y[test]

    y_score = model.predict([X1_train[test],X2_train[test]])

    fpr, tpr, _= roc_curve(y_test,  y_score)
    auc1 = metrics.roc_auc_score(y_test, y_score)
    
    dnn_auc_list.append(auc1)


    
    y_score=y_score[:,0]
    
    for i in range(0,len(y_score)):
        if(y_score[i]>0.5):
            y_score[i]=1
        else:
            y_score[i]=0
            
    cm1=confusion_matrix(y[test][:,0],y_score)
    acc1 = accuracy_score(y[test][:,0], y_score, sample_weight=None)
    spec1= (cm1[0,0])/(cm1[0,0]+cm1[0,1])
    sens1 = recall_score(y[test][:,0], y_score, sample_weight=None)
    prec1=precision_score(y[test][:,0], y_score, sample_weight=None)
    f11=f1_score(y[test][:,0], y_score)
    ka1=cohen_kappa_score(y[test][:,0], y_score)

    sensitivity1.append(sens1)
    specificity1.append(spec1)
    accuracy1.append(acc1)
    precision1.append(prec1)
    F11.append(f11)
    kappa1.append(ka1)

    coef=matthews_corrcoef(y[test], y_score, sample_weight=None)
    m_coef.append(coef)
    precision12, recall12, thresholds_AUPR = precision_recall_curve(y[test][:,0],y_score)
    AUPR = auc(recall12, precision12)
    aupr_list.append(AUPR)

    if acc1>max_accuracy:
        max_accuracy=acc1
        dnn_fpr=fpr[:]
        dnn_tpr=tpr[:]

dnn_fpr=pd.DataFrame(dnn_fpr)
dnn_tpr=pd.DataFrame(dnn_tpr)

##########################创建保存文件##############################
import sys  # 需要引入的包
 
# 以下为包装好的 Logger 类的定义
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        # self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

#########################保存以下信息###############################
sys.stdout = Logger('/No.2/test2 net/HV.txt')
mean_acc1 = np.mean(accuracy1)
mean_acc1 = np.mean(accuracy1)
std_acc1 = np.std(accuracy1)
var_acc1 = np.var(accuracy1)
print("--Accuracy1:" + str(mean_acc1) + " Â± " + str(std_acc1))
print("Accuracy_Var:" + str(mean_acc1) + " Â± " + str(var_acc1))

mean_spec1 = np.mean(specificity1)
std_spec1 = np.std(specificity1)
print("Specificity1:" + str(mean_spec1) + " Â± " + str(std_spec1))

mean_sens1 = np.mean(sensitivity1)
std_sens1 = np.std(sensitivity1)
print("Sensitivity1:" + str(mean_sens1) + " Â± " + str(std_sens1))

mean_prec1 = np.mean(precision1)
std_prec1 = np.std(precision1)
print("Precison1:" + str(mean_prec1) + " Â± " + str(std_prec1))

mean_F11=np.mean(F11)
std_F11=np.std(F11)
print("F1:"+str(mean_F11)+" Â± "+str(std_F11))

mean_kappa1=np.mean(kappa1)
std_kappa1=np.std(kappa1)
print("Kappa1:"+str(mean_kappa1)+" Â± "+str(std_kappa1))

mean_coef = np.mean(m_coef)
std_coef = np.std(m_coef)
print("MCC1:" + str(mean_coef) + " Â± " + str(std_coef))

print("AUPR1:" + str(np.mean(aupr_list)))

print("----dnn_auc_list:" + str(dnn_auc_list))
print("AUC1:" + str(np.mean(dnn_auc_list)))

end1 = time.time()
end11 = end1 - start
print(f"Runtime of the program is {end1 - start}")