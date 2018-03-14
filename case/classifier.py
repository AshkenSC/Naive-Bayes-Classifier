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

# load data.npy and target.npy
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

# calculate text vector sum of every category
vecSum = [[1]*len(trainCountsArray[0]) for i in range(len(vecMap))]     # initialize sum array
for i in range(len(vecMap)):
    for j in range(len(vecMap[i])):
        vecSum[i] += vecMap[i][j].textVec

# calculate category condition possibility according to vecSum[i]/categoryWordCount[i]
categoryWordCount = np.ones(len(vecSum))   # categoryWordCount[i]: word number category i
for i in range(len(vecSum)):
    for j in range(len(vecSum[i])):
        categoryWordCount[i] += vecSum[i][j]

    # condition possibility of every category
conditionPossibility = [[0]*len(trainCountsArray[0]) for i in range(len(vecSum))]
for i in range(6):
    for j in range(len(vecSum[i])):
        conditionPossibility[i][j] = np.log(vecSum[i][j] / categoryWordCount[i])     # use log to smooth

    # overall possibility of every category
categoryPossibility = np.zeros(len(vecSum))
for i in range(len(vecSum)):
    for j in range(len(vecMap[i])):
        categoryPossibility[i] += 1
categoryPossibility /= len(trainCountsArray)


# draw graph for different categories
fig = plt.figure()
for i in range(len(vecSum)):
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(np.arange(0, len(conditionPossibility[0])),
               np.array(conditionPossibility[i])*categoryPossibility[i],
               label=i,
               alpha=0.3)
    ax.legend()
plt.show()




