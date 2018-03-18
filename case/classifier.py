# encoding=utf-8
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm  import  LinearSVC
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
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

# step 1 of vectorizing method: count vectorizer (bag of words)
from sklearn.feature_extraction.text import CountVectorizer

# TODO: Modify max_df and min_df, observe variation of precision, recall and f1-score along with the change, and make a graph
countVector = CountVectorizer(stop_words=stopWords, decode_error='ignore', max_df=0.5, min_df=0.0005)
trainCounts = countVector.fit_transform(data)
# shape output format: (sample number, dict size)
print("words freq shape:", trainCounts.shape)

# step 2 of vectorize method: TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfTransformer
tfTransformer = TfidfTransformer(use_idf=True).fit(trainCounts)
trainTF = tfTransformer.transform(trainCounts)
print("TF-IDF shape:", trainTF.shape)

# build naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
isTF_IDF = 1     # a flag to select character
if (isTF_IDF == 0):
    # 1) use bag of words vector
    trainVector = trainCounts
else:
    # 2) use TF-IDF vetor
    trainVector = trainTF

# K-fold validation to split train set and test set
# TODO: Update to K-fold cross validation
# TODO: 只划分了数据集，没有划分对应分类结果集
kf = KFold(n_splits=10, shuffle=False)
# kf.split() returns index values of train and test after split.
kf.split(trainVector)

# TODO: Set different alpha(for Laplace/Lidstone smoothing)
naiveBayesClassifier = MultinomialNB(alpha=0.2).fit(trainSet, target)

# make prediction
# the parameter of predict(data) is the test dataset
# (obseleted) predicted = naiveBayesClassifier.predict(tfTransformer.transform(countVector.transform(data)))
# TODO: predicted = naiveBayesClassifier.predict(testSet)

# print test results
from sklearn import metrics
print(metrics.classification_report(target, predicted))

# --------------
# Reinvent NB classifier wheel for probability distribution graph
def NBwheel():
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

# --------------
# draw graph for different categories
# Before using this function, call NBwheel() first
def DrawCtgDistribution():
    fig = plt.figure()
    for i in range(len(vecSum)):
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(np.arange(0, len(conditionPossibility[0])),
                   np.array(conditionPossibility[i])*categoryPossibility[i],
                   label=i,
                   alpha=0.3)
        ax.legend()
    plt.show()

# --------------
# draw t-SNE graph
# Before using this function, call NBwheel() first
# TODO: Convert sparse matrix to dense matrix
def DrawTSNE():
    X_tsne = TSNE(learning_rate=100).fit_transform(trainCounts.toarray())
    X_pca = PCA().fit_transform(trainCounts.toarray())

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target)
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target)
    plt.show()




