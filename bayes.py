from numpy import *
import re


# load training text list and its classification list
def loadDataSet():
    postingList = [[' my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', ' to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# create vocabulary set according to the training set
def createVocabList(dataSet):
    # create an empty set
    vocabSet = set([])
    for document in dataSet:
        # create union set
        vocabSet = vocabSet | set(document)

    # transfer the orderless vacabulary set to a ordered vocabulary list
    vocabList = list(vocabSet)
    vocabList.sort()

    return vocabList


# create sentence vector according to sentence and vocab list
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("The word: %s is not in my Vocabulary." % word)
    return returnVec


# training function that calculate conditional probabilities p(w|c)
# in p(w|c), w stands for "word" and c stands for "classification"
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # initialize probability
    p0Num = ones(numWords);
    p1Num = ones(numWords)
    p0Denom = 2.0;
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # add up vectors
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # !!!Due to extra indent, the following two lines are inside the for cycle,!!!
    # !!!which leads to wrong result!!!

    p1Vect = log(p1Num / p1Denom)  # change to log()
    # do division on every element
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive


# classify sentence vector according to results of training function
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # multiply elements
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# parse text
def textParse(bigString):
    listOfTokens = re.split(r'\W', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]




