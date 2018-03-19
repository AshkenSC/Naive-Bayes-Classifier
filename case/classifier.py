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

# Toggle array display integrity
# np.set_printoptions(threshold=np.inf)

os.path.abspath('..')

stopWords = ['的', '等', '了', '并', '得', '等', '而且', '个', '和', '还是', '还有', '有', '以']

# Load data.npy and target.npy
data = sp.load('data.npy')
target = sp.load('target.npy')

# Step 1 of vectorizing method: count vectorizer (bag of words)
from sklearn.feature_extraction.text import CountVectorizer

# TODO: Modify max_df and min_df, observe variation of precision, recall and f1-score along with the change, and make a graph
countVector = CountVectorizer(stop_words=stopWords, decode_error='ignore')
trainCounts = countVector.fit_transform(data)
# Shape output format: (sample number, dict size)
print("words freq shape:", trainCounts.shape)

# Step 2 of vectorize method: TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfTransformer
tfTransformer = TfidfTransformer(use_idf=True).fit(trainCounts)
trainTF = tfTransformer.transform(trainCounts)
print("TF-IDF shape:", trainTF.shape)

# Build naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

isTF_IDF = 1     # a flag to select character
if (isTF_IDF == 0):
    # 1) Use bag of words vector
    trainVector = trainCounts
else:
    # 2) Use TF-IDF vetor
    trainVector = trainTF

# K-fold validation to split train set and test set
kf = KFold(n_splits=10, shuffle=False)
# kf.split() returns index values of train and test after split.
kf.get_n_splits(trainVector)
print("Validation method: " , kf)

# Repeat test for K times, where K is now 10
precision, recall, f1Score, support = [], [], [], []
testCount = 1;  # Use this to count test when output result
for trainIndex,testIndex in kf.split(trainVector):

    # Data set and target set are split TOGETHER using trainIndex and testIndex as their COMMON index number
    splitTrainData, splitTestData = trainVector[trainIndex], trainVector[testIndex]
    splitTrainTarget, splitTestTarget = target[trainIndex], target[testIndex]

    # TODO: Set different Classifier model(0.Bernoulli, 1.Multinomial)
    # TODO: Set different alpha(for Laplace/Lidstone smoothing)
    classifierType = 0
    if classifierType == 0:
        naiveBayesClassifier = BernoulliNB(alpha=1.0).fit(splitTrainData, splitTrainTarget)
    elif classifierType == 1:
        naiveBayesClassifier = MultinomialNB(alpha=1.0).fit(splitTrainData, splitTrainTarget)

    # Make prediction. The parameter of predict(data) is the test dataset
    predicted = naiveBayesClassifier.predict(splitTestData)

    # Get test result statistics
    from sklearn import metrics
    # Store statistics into lists respectively
    precision.append(metrics.precision_recall_fscore_support(splitTestTarget, predicted)[0])
    recall.append(metrics.precision_recall_fscore_support(splitTestTarget, predicted)[1])
    f1Score.append(metrics.precision_recall_fscore_support(splitTestTarget, predicted)[2])
    support.append(metrics.precision_recall_fscore_support(splitTestTarget, predicted)[3])

    # Print report
    '''
    print("***** Test No.", testCount, "*****")
    testCount += 1
    report = metrics.classification_report(splitTestTarget, predicted)
    print(report)
    '''

# --------------
# Reinvent NB classifier wheel for probability distribution graph
def NBwheel():
    # Get train set count vector 2d array
    trainCountsArray = trainCounts.toarray()

    # Get train set TF-IDF 2d array
    tfIdfArray = trainTF.toarray()

    # Build a class to store mapping between text vetor and its category
    class VecCategoty:
        def __init__(self):
            textVec = []
            category = 0
    vecMap = [[],[],[],[],[],[]]

    # Store the text-category mapping into an array, vecMap[i][j] means the text is No.j in category No.i
    for i in range(len(trainCountsArray)):
        vecTemp = VecCategoty()
        vecTemp.textVec = trainCountsArray[i]
        vecTemp.category = target[i]
        vecMap[vecTemp.category].append(vecTemp)

    # Calculate text vector sum of every category
    vecSum = [[1]*len(trainCountsArray[0]) for i in range(len(vecMap))]     # initialize sum array
    for i in range(len(vecMap)):
        for j in range(len(vecMap[i])):
            vecSum[i] += vecMap[i][j].textVec

    # Calculate category condition possibility according to vecSum[i]/categoryWordCount[i]
    categoryWordCount = np.ones(len(vecSum))   # categoryWordCount[i]: word number category i
    for i in range(len(vecSum)):
        for j in range(len(vecSum[i])):
            categoryWordCount[i] += vecSum[i][j]

        # Condition possibility of every category
    conditionPossibility = [[0]*len(trainCountsArray[0]) for i in range(len(vecSum))]
    for i in range(6):
        for j in range(len(vecSum[i])):
            conditionPossibility[i][j] = np.log(vecSum[i][j] / categoryWordCount[i])     # use log to smooth

        # Overall possibility of every category
    categoryPossibility = np.zeros(len(vecSum))
    for i in range(len(vecSum)):
        for j in range(len(vecMap[i])):
            categoryPossibility[i] += 1
    categoryPossibility /= len(trainCountsArray)

# --------------
# Draw graph for different categories
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
# Draw t-SNE graph
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

# --------------
# Draw straight line charts (for Bernoulli VS Multinomial)
def DrawLineChart(dataList):
    fig = plt.figure(dpi=128, figsize=(10, 6))

    # Draw plot
    plt.plot(dataList, marker='D', alpha=0.9)

    # Set format
    plt.ylim(0.75, 1.04)
    plt.legend(['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6'], loc='upper left')
    plt.title('F1-score of Different Text Types in Bernoulli Model')
    #plt.title('F1-score of Different Text Types in Multinomial Model')

    plt.show()

# --------------
# TODO: Draw smooth line charts (for alpha and max_df&min_df)


print(f1Score)
DrawLineChart(f1Score)



