import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存，按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
#session = tf.compat.v1.Session(config=config)
import numpy as np
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn import metrics
from keras.optimizer_v2.adam import Adam
from keras.layers import Input, Conv1D, LSTM, GRU, Bidirectional, MaxPooling1D, Conv2D, Add, Permute, \
    multiply, AveragePooling2D, AveragePooling1D, GlobalAveragePooling1D, GlobalAveragePooling2D, Dropout, Flatten, \
    Dense, BatchNormalization, Activation, Concatenate, Reshape, Multiply, GlobalMaxPooling2D, Lambda, AveragePooling1D
from keras.models import Model, load_model
from keras.regularizers import l1, l2
from keras.utils.np_utils import to_categorical
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold  # k折叠交叉验证
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")


# 性能评价指标
def show_performance(y_true, y_pred):
    # 定义tp, fp, tn, fn初始值
    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] >= 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] >= 0.5:
                FP += 1
            else:
                TN += 1

    # 计算敏感性Sn
    Sn = TP / (TP + FN + 1e-06)
    # 计算特异性Sp
    Sp = TN / (FP + TN + 1e-06)
    # 计算Acc值
    Acc = (TP + TN) / len(y_true)
    # 计算MCC：马修斯相关系数是在混淆矩阵环境中建立二元分类器预测质量的最具信息性的单一分数
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    return Sn, Sp, Acc, MCC


def acc_loss_plot(train_loss, train_acc, val_loss, val_acc, fold_count=0):
    with plt.style.context(['wugenqiang', 'grid', 'no-latex']):
        # 创建一个图
        plt.figure()
        plt.plot(train_loss, label='train loss')
        plt.plot(train_acc, label='train acc')
        plt.plot(val_loss, label='val loss')
        plt.plot(val_acc, label='val acc')
        # plt.grid(True, linestyle='--', alpha=0.5)  # 增加网格显示
        plt.title('acc-loss')  # 标题
        plt.xlabel('epoch')  # 给x轴加注释
        plt.ylabel('acc-loss')  # 给y轴加注释
        plt.autoscale(tight=True)  # 自动缩放(紧密)
        plt.legend(loc="upper right")  # 设置图例显示位置
        if fold_count == 0:
            plt.savefig('images/acc-loss.jpg', dpi=600, bbox_inches='tight')  # bbox_inches可完整显示
        else:
            plt.savefig('images/acc-loss_' + str(fold_count) + '_fold.jpg', dpi=600, bbox_inches='tight')
        plt.show()


def plot_metric(history, metric, fold_count=0):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)

    with plt.style.context(['wugenqiang', 'grid', 'no-latex']):
        plt.plot(epochs, train_metrics, 'b--')
        plt.plot(epochs, val_metrics, 'r-')
        plt.title('Training and validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, 'val_' + metric])
        if fold_count == 0:
            plt.savefig('./images/' + metric + '.jpg', dpi=600, bbox_inches='tight')
        else:
            plt.savefig('./images/' + metric + '_' + str(fold_count) + '_fold.jpg', dpi=600, bbox_inches='tight')
        plt.show()


def print_performance(performance, row):
    print('Sn = %.2f%% ± %.2f%%' % (np.mean(performance[:, row, 0]) * 100, np.std(performance[:, row, 0]) * 100))
    print('Sp = %.2f%% ± %.2f%%' % (np.mean(performance[:, row, 1]) * 100, np.std(performance[:, row, 1]) * 100))
    print('Acc = %.2f%% ± %.2f%%' % (np.mean(performance[:, row, 2]) * 100, np.std(performance[:, row, 2]) * 100))
    print('MCC = %.4f ± %.4f' % (np.mean(performance[:, row, 3]), np.std(performance[:, row, 3])))
    print('AUC = %.4f ± %.4f' % (np.mean(performance[:, row, 4]), np.std(performance[:, row, 4])))


def SNO_DCA(train, label):
    # 10折叠交叉验证
    k_fold = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

    # 定义10个矩阵，1行5列全0矩阵
    performance = np.zeros((10, 1, 5))

    # 用于存放训练集的预测
    train_y_pred = np.zeros(len(train))

    all_Sn = []
    all_Sp = []
    all_Acc = []
    all_MCC = []
    all_AUC = []
    print('10折叠交叉验证: ')
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    for fold_, (train_index, validation_index) in enumerate(k_fold.split(train, label)):
        print("fold {} times".format(fold_ + 1))
        # 训练集、训练集标签
        X_train, y_train = train[train_index], label[train_index]
        # 验证集、验证集标签
        X_validation, y_validation = train[validation_index], label[validation_index]

        y_train = to_categorical(y_train, num_classes=2)
        y_validation = to_categorical(y_validation, num_classes=2)


        # 载入模型
        model = load_model('train model/SNO-DCA_model_{}_fold.h5'.format(fold_ + 1))

        # 预测值
        y_pred = model.predict(X_validation, verbose=0)

        # 存放训练集的预测
        train_y_pred[validation_index] = y_pred[:, 1]
        from sklearn.metrics import roc_auc_score
        # 获取性能评价Sn, Sp, Acc, MCC, AUC
        Sn, Sp, Acc, MCC = show_performance(y_validation[:, 1], y_pred[:, 1])
        AUC = roc_auc_score(y_validation[:, 1], y_pred[:, 1])

        performance[fold_, 0, :] = np.array((Sn, Sp, Acc, MCC, AUC))
        print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))
        all_Sn.append(Sn)
        all_Sp.append(Sp)
        all_Acc.append(Acc)
        all_MCC.append(MCC)
        all_AUC.append(AUC)
    print('SNO-DCA: ')
    print_performance(performance, 0)
    best_MCC = all_MCC.index(np.max(all_MCC))
    return train_y_pred, best_MCC

if __name__ == '__main__':
    # 设置随机种子，很nice
    np.random.seed(0)
    tf.random.set_seed(1)  # for reproducibility

    # 超参数设置
    WINDOWS = 41  # 序列窗口设置

    f_r_train = open("data/SNO-train-shuffle.txt", "r", encoding='utf-8')
    f_r_test = open("data/SNO-test-shuffle.txt", "r", encoding='utf-8')

    # 训练序列片段构建
    train_data = f_r_train.readlines()
    # 预测序列片段构建
    test_data = f_r_test.readlines()

    # 关闭文件
    f_r_train.close()
    f_r_test.close()

    # 数据编码
    import encoding

    # one_hot编码序列片段
    train_X_1, train_Y = encoding.one_hot(train_data, windows=WINDOWS)
    print(train_X_1.shape)
    train_Y = to_categorical(train_Y, num_classes=2)
    test_X_1, test_Y = encoding.one_hot(test_data, windows=WINDOWS)
    test_Y = to_categorical(test_Y, num_classes=2)
    #print(test_Y)

    train_y_pred, best_MCC = SNO_DCA(train_X_1, train_Y[:, 1])
    # print(train_y_pred)

    # del model  # 删除现有模型

    # 载入模型
    model = load_model('test model/SNO-DCA_model.h5')
    # 预测值
    test_pred = model.predict([test_X_1], verbose=0)
    # 预测值保存入文件
    pd.DataFrame(test_pred).to_csv('data/test_label_pred.csv', index=False)

    # 验证预测结果
    # 获取性能评价Sn, Sp, Acc, MCC, AUC
    print('-----------------------------------------------test---------------------------------------')
    Sn, Sp, Acc, MCC = show_performance(test_Y[:, 1], test_pred[:, 1])
    AUC = roc_auc_score(test_Y[:, 1], test_pred[:, 1])
    print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f' % (Sn, Sp, Acc, MCC, AUC))
""""
    from sklearn.metrics import roc_curve,auc

    #fpr, tpr, thersholds = roc_curve(train_Y[:, 1], train_y_pred, pos_label=1)
    fpr, tpr, thersholds = roc_curve(test_Y[:,1], test_pred[:,1], pos_label=1)
    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))
    roc_auc = auc(fpr, tpr)
    #plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    plt.plot(fpr, tpr, color='darkorange', label='ROC (area = {0:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('snoroc-train.jpg', dpi=800)
    plt.show()
"""
