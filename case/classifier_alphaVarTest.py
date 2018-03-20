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

countVector = CountVectorizer(stop_words=stopWords, decode_error='ignore')
trainCounts = countVector.fit_transform(data)
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
kf.get_n_splits(trainVector)
print("Validation method: " , kf)

# Variables for alpha test
f1Variation = []        # A N*6 array that stores f1-score variation of 6 categories(N is how many times alpha changes)
f1SubList = []          # A 10*6 TEMP array that stores f1-score of 10-fold tests under certain alpha value
graph_xAxis = []        # A list for x-axis ticker of the graph
alphaScale = 50         # 1/alphaScale is delta of the changing alpha

# In every outer loop alphaScale varies
for i in range(alphaScale):
    # Repeat for K times where K is now 10, do K-fold cross validation
    for trainIndex,testIndex in kf.split(trainVector):

        # Data set and target set are split TOGETHER using trainIndex and testIndex as their COMMON index number
        splitTrainData, splitTestData = trainVector[trainIndex], trainVector[testIndex]
        splitTrainTarget, splitTestTarget = target[trainIndex], target[testIndex]

        # TODO: Set different alpha(for Laplace/Lidstone smoothing)
        classifierType = 1
        if classifierType == 0:
            naiveBayesClassifier = BernoulliNB(alpha=1.0/alphaScale * i).fit(splitTrainData, splitTrainTarget)
        elif classifierType == 1:
            naiveBayesClassifier = MultinomialNB(alpha=1.0/alphaScale * i).fit(splitTrainData, splitTrainTarget)

        # Make prediction. The parameter of predict(data) is the test dataset
        predicted = naiveBayesClassifier.predict(splitTestData)

        # Get test result statistics
        from sklearn import metrics
        # Store statistics into lists respectively
        f1SubList.append(metrics.precision_recall_fscore_support(splitTestTarget, predicted)[2])

    # Calculate arithmetic average of F1-score for every category under this ALPHA value
    f1Variation.append(np.sum(f1SubList, axis=0) / 10)
    f1SubList = []  # Clear the temp list

    # Update x-axis ticker list
    graph_xAxis.append(1.0 / alphaScale * i)

# test print
print('lenf1var', len(f1Variation))
print('lenf1var0', len(f1Variation[0]))
print(f1Variation)

# --------------
# Draw line chart for alpha variation
def DrawAlphaVariation(dataList):
    fig = plt.figure(dpi=128, figsize=(10, 6))

    # Draw plot
    plt.plot(graph_xAxis, dataList, alpha=0.9)

    # Set other format
    plt.grid(alpha=0.3)
    plt.locator_params('x', nbins=20)
    plt.locator_params('y', nbins=20)
    #plt.ylim(0.75, 1.0)
    plt.legend(['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6'], loc='lower right')
    plt.xlabel('alpha')
    plt.ylabel('F1-score')
    plt.title('Variation of F1-score Along Variation of Alpha')

    plt.show()

# Execute draw function
DrawAlphaVariation(f1Variation)



