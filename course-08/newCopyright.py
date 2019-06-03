import numpy as np
import pandas as pd
import random


import csv
import sys
import os
import jieba
import re
import pickle

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score,cross_validate # 交叉验证所需的函数
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score,f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot  as plt


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import  GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def load_data():
    data_path = '/Users/heany/code/AI-For-NLP/dataset/sqlResult_1558435.csv'
    content = pd.read_csv(data_path, encoding='gb18030')

    idx_pos = content[content['source'] == '新华社']['id'].tolist()
    idx_neg = content[content['source'] != '新华社']['id'].tolist()

    data = pd.DataFrame()
    data['id'] = content['id']
    data['label'] = pd.Series([1 if idx in idx_pos else 0 for idx in content['id']])

    # idx_test_neg = random.sample(idx_neg, 950)
    # idx_test_pos = random.sample(idx_pos, 9050)

    #=========================== split the train data and test data ===========
    # train_x, test_x, train_y, test_y = train_test_split(content['content'],data['label'],test_size=0.12,random_state=0)
    # return train_x, test_x, train_y, test_y
    return content['content'],data['label']


def token(string):
    return re.findall(r'[\d|\w]+', str(string))

def cut(string):
    return ' '.join(jieba.cut(string))

def Preprocess_token(content):
    news_content = [token(n) for n in content]
    news_content = [' '.join(x) for x in news_content]
    news_content = [cut(y) for y in news_content]
    return news_content


def extract_feature(data_x, data_y):
    data_x_p = Preprocess_token(data_x.tolist())

    vectorized = TfidfVectorizer(max_features=10000)

    x_data = vectorized.fit_transform(data_x_p)
    y_data = np.array(data_y)

    # save the feature
    with open('./feature.pickle', 'wb') as f:
        pickle.dump([x_data, y_data], f)
    return x_data, y_data

# +++++++++++++++++ ROC Curve ++++++++++++++++
def draw_roc_auc(y_test, y_pred):
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

def main():
    print('----------  load data ')
    # data_x, data_y = load_data()
    #
    # x_data, y_data = extract_feature(data_x,data_y)


    print('----------  load the feature')
    with open('./feature.pickle', 'rb') as f:
        x_data, y_data = pickle.load(f)

    # split the data into two parts
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.12,
                                                        random_state=2)

    print('-----------  model fit ')
    # acc:0.9773  f1:0.9872
    # clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(train_x, train_y)

    #acc: 0.8938  f1:0.9362
    # clf = GaussianNB().fit(train_x.todense(),train_y)

    #acc:0.8786  f1:0.9354
    # clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(train_x, train_y)

    #acc:0.9893    f1:0.9939
    clf = DecisionTreeClassifier(random_state=1).fit(train_x, train_y)

    # print(cross_val_score(clf, test_x, test_y, cv=10))

    print('-------------- model predict')
    y_pred = clf.predict(test_x.todense())

    print("accuracy_score:{}".format(accuracy_score(test_y, y_pred)))
    print("f1-score:{}".format(f1_score(test_y, y_pred, average='binary')))

    # draw_roc_auc(test_y,y_pred)
    print('y_true:{}'.format(sum(test_y.tolist())))
    print('y_pred:{}'.format(sum(y_pred.tolist())))


if __name__ == '__main__':
    main()
