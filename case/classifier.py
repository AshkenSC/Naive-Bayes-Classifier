# encoding=utf-8
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from  sklearn.svm  import  LinearSVC
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import os

# toggle array display integrity
# np.set_printoptions(threshold=np.inf)

os.path.abspath('..')

stopWords = ['的', '等', '了', '并', '得', '等', '而且', '个', '和', '还是', '还有', '有', '以']

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

# test load data.npy and target.npy
data = sp.load('data.npy')
target = sp.load('target.npy')

# vectorize method step1: count vectorizer (bag of words)
from sklearn.feature_extraction.text import CountVectorizer

countVector = CountVectorizer(stop_words=stopWords, decode_error='ignore')
trainCounts = countVector.fit_transform(data)
# .shape output format: (sample number, dict size)
# print(trainCounts.shape)

# vectorize method step2: TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfTransformer
tfTransformer = TfidfTransformer(use_idf=False).fit(trainCounts)
trainTF = tfTransformer.transform(trainCounts)
# print(trainTF.shape)

# build naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
    # 1) use bag of words vector
naiveBayesClassifier = MultinomialNB().fit(trainCounts, target)
    # 2) use TF-IDF vetor
# naiveBayesClassifier = MultinomialNB().fit(trainTF, target)

# print test results
predicted = naiveBayesClassifier.predict(tfTransformer.transform(countVector.transform(data)))

from sklearn import metrics
# print(metrics.classification_report(target, predicted))

# get train set count vector 2d array
trainCountsArray = trainCounts.toarray()

# get train set TF-IDF 2d array
tfIdfArray = trainTF.toarray()


# build a class to store mapping between text vetor and its category
class VecCategoty:
    def __init__(self):
        textVec = []
        category = 0
vecMap = [[],[],[],[],[],[]]

# store the text-category mapping into an array, vecMap[i][j] means the text is No.j in category No.i
for i in range(len(trainCountsArray)):
    vecTemp = VecCategoty()
    vecTemp.textVec = trainCountsArray[i]
    vecTemp.category = target[i]
    vecMap[vecTemp.category].append(vecTemp)

# calculte word count sum of every category
vecSum = [[0]*len(trainCountsArray[0]) for i in range(len(vecMap))]     # initialize sum array

# convert ca

for i in range(len(vecMap)):
    for j in range(len(vecMap[i])):
        vecSum[i] += vecMap[i][j].textVec

# draw graph for different categories
fig = plt.figure()
for i in range(len(vecSum)):
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(np.arange(0, len(trainCountsArray[0])),
               vecSum[i],
               label=i,
               alpha=0.3)
    ax.legend()
plt.show()





