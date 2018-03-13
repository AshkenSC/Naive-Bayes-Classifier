import matplotlib.pyplot as plt
from bayes import *

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


# a function that finds spams in mailbox
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # import and analyze text file

        # these are "bad" emails in training set
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # these are "good" emails in training set
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50));
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
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    # calculate error rate
    errorRate = float(errorCount) / len(testSet)
    print("the error rate is: %f" % (errorRate))

    # draw scattergraph
    '''
    parameters of scatter():
    1) arange(0, len(p0V)): x value range of graph
        where len(p0V) actually returns the length of vocaulary set
    2) p0V * (1-pSpam): 
        p0V is the vetor(array) that stores P(vocabulary | docType0)
        1-pSpam is P(ham), the general probability of a mail being ham(good mail)
        
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(arange(0, len(p0V)),
               p0V * (1-pSpam),
               label='ham',
               alpha=0.3)
    ax.legend()
    bx = fig.add_subplot(1, 1, 1)
    bx.scatter(arange(0, len(p1V)),
               p0V * pSpam,
               label='spam',
               alpha=0.3)
    bx.legend()
    plt.show()

# execute test function
# spamTest()


