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
countVector = CountVectorizer(stop_words=stopWords, decode_error='ignore', max_df=0.5, min_df=10)
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

    print("***** Test No.", testCount, "*****")
    testCount += 1
    report = metrics.classification_report(splitTestTarget, predicted)
    print(report)


# --------------
# Draw line charts for Bernoulli VS Multinomial for comparison
def DrawModelComparison(dataList):
    fig = plt.figure(dpi=128, figsize=(10, 6))

    # Draw plot
    plt.plot(dataList, marker='D', alpha=0.9)

    # Set format
    if classifierType == 0:
        plt.title('F1-score of Different Text Types in Bernoulli Model')
    else:
        plt.title('F1-score of Different Text Types in Multinomial Model')
    plt.ylim(0.3, 1.04)
    plt.grid(alpha=0.5)
    plt.xlabel('test set serial number')
    plt.ylabel('F1-score')
    plt.legend(['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6'], loc='lower right')

    plt.show()

# Execute draw function
print(f1Score)
DrawModelComparison(f1Score)
