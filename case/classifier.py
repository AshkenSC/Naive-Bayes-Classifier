# encoding=utf-8
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from  sklearn.svm  import  LinearSVC
import scipy as sp
import numpy as np
import os

os.path.abspath('..')

list = ['的', '等']

def twice_classify(first_class):

    count = [[0 for i in range(10)]for j in range(7)]

    os.path.abspath('..')

    path_1 = "../train_"
    path = path_1 + str(first_class)

    #读取加载数据
    data_reviews = load_files(path, encoding = "GBK")  # 训练集

    test_review = load_files("../test", encoding = "GBK")  # 测试集

    # 读取
    data = data_reviews.data  # 训练集
    target = data_reviews.target

    test_data = test_review.data  # 测试集
    test_target = test_review.target

    # 初始化TfidfVectorizer
    count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore', stop_words = list)

    # 加载训练集，数据全部用于训练
    data_train, data_test_zero, target_train, target_test_zero = train_test_split(data, target, test_size = 0)
    tf_train = count_vec.fit_transform(data_train)  # 训练集tf-idf

    # 加载测试集，数据全部用于测试
    test_train_zero, data_test, target_train_zero, target_test = train_test_split(test_data, test_target,
                                                                                  test_size = 0.999)
    tf_test = count_vec.transform(data_test)  # 测试集tf-idf

    # 调用MultinomialNB分类器
    clf = MultinomialNB().fit(tf_train, target_train)
    class_predicted = clf.predict(tf_test)

    return int(class_predicted + 1)


def first_classify(my_str):
    with open("../test/1/test.txt", 'w') as file_project:
        file_project.write(my_str)
        file_project.close()
    test_review = load_files("../test", encoding = "GBK")

    # 读取
    data = sp.load('data.npy')  # 训练集
    target = sp.load('target.npy')

    test_data = test_review.data  # 测试集
    test_target = test_review.target

    count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore', stop_words = list)

    data_train, data_test_zero, target_train, target_test_zero = train_test_split(data, target, test_size = 0)
    tf_train = count_vec.fit_transform(data_train)  # 训练集tf-idf

    test_train_zero, data_test, target_train_zero, target_test = train_test_split(test_data, test_target,
                                                                                  test_size = 0.999)
    tf_test = count_vec.transform(data_test)  # 测试集tf-idf

    clf = LinearSVC(random_state = 0).fit(tf_train, target_train)
    class_predicted = clf.predict(tf_test)

    return int(class_predicted+1)

# test what's in the data.npy and target.npy
data = sp.load('data.npy')
target = sp.load('target.npy')
'''
for i in range(20):
    print("length of data.npy is %d, and length of target.npy is %d." % (len(data), len(target)))
    print("Case number %d, “%s” belongs to %d" % (i+1, data[i], target[i]))
'''
classes = []
for i in range(10):
    classes.append(0)
for i in range(len(target)):
    for j in range(10):
        if(j == target[i]):
            classes[j] = classes[j]+1
        else:
            continue
for i in range(len(classes)):
    if(classes[i] != 0):
        print("class No.%d has %d cases. " % (i+1, classes[i]))