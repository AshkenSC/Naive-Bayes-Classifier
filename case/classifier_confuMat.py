import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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
classNames = ['1', '2', '3', '4', '5', '6']

# Step 1 of vectorizing method: count vectorizer (bag of words)
from sklearn.feature_extraction.text import CountVectorizer

# TODO: Modify max_df and min_df, observe variation of precision, recall and f1-score along with the change, and make a graph
countVector = CountVectorizer(stop_words=stopWords, decode_error='ignore', max_df=0.5, min_df=8)
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

isTF_IDF = 0     # a flag to select character
if isTF_IDF == 0:
    # 1) Use bag of words vector
    trainVector = trainCounts
else:
    # 2) Use TF-IDF vetor
    trainVector = trainTF

# Split the data into a training set and a test set
splitTrainData, splitTestData, splitTrainTarget, splitTestTarget = train_test_split(trainVector, target, random_state=0)

# TODO: Set different Classifier model(0.Bernoulli, 1.Multinomial)
classifierType = 0
if classifierType == 0:
    naiveBayesClassifier = BernoulliNB(alpha=0.2).fit(splitTrainData, splitTrainTarget)
elif classifierType == 1:
    naiveBayesClassifier = MultinomialNB(alpha=0.2).fit(splitTrainData, splitTrainTarget)

# Make prediction. The parameter of predict(data) is the test dataset
predicted = naiveBayesClassifier.predict(splitTestData)

# Print report
from sklearn import metrics
report = metrics.classification_report(splitTestTarget, predicted)
print(report)

# Draw confusion matrix
def DrawConfusionMatrix(cm, classes,
                        normalize=False,
                        title='Confusion Matrix',
                        cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=15)

    plt.ylabel('True doc type', fontsize=15)
    plt.xlabel('Predicted doc type', fontsize=15)


# Compute confusion matrix
cnf_matrix = confusion_matrix(splitTestTarget, predicted)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(dpi=128, figsize=(8, 7))
DrawConfusionMatrix(cnf_matrix, classes=classNames,
                    title='Confusion Matrix, without Normalization')

# Plot normalized confusion matrix
plt.figure(dpi=128, figsize=(8, 7))
DrawConfusionMatrix(cnf_matrix, classes=classNames, normalize=True,
                    title='Normalized Confusion Matrix')

plt.show()

