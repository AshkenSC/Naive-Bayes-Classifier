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


# clean spam in text
def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        # import and analyze text file
        # TODO: set text path
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fulltext.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50);
    testSet = []

    # randomly build training set
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0

    # classify test set
    wordVector = setOfWords2Vec(vocabList, docList[docIndex])
    if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
        errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))

# a convinience function, in order to test classifier
def testingNB():
    listsOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listsOPosts)
    trainMat = []
    for postinDoc in listsOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)

    # print vocabulary list, overall abusive probability, and condition probability when the class is given
    print('Vocabulary list is:', myVocabList)
    print(pAb)
    print(p0V)
    print(p1V)

    # test entry #1
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    # test entry #2
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    # test entry #3
    testEntry = ['i', 'love', 'you', 'you', 'are', 'such', 'a', 'stupid', 'monster', 'asshole']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    # test entry #4
    testEntry = ['please', 'thank', 'you', 'very', 'much']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


testingNB()
