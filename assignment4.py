import json
import dynet as dy
import numpy as np
import random
import math

bigramArray=[]
uniquePoem=[]
oneHotVectorDict = {}
oneHotVectorArray = []
totalList = []
totalBigram = 0
SIZE = 100

#creates language model
def languagemodel(s, n):
    s = s.lower()
    tokens = [token for token in s.split(" ") if token != ""]
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

#reads json file , add start end tokens ,create language model
def readJSON():
    global uniquePoem
    global totalList
    with open('unim_poem.json') as json_file:
        data = json.load(json_file)
        for i in range(SIZE):#len(data)
            eachPoem = data[i]["poem"]
            eachPoem = str(eachPoem).replace("\n"," \n ")
            eachPoem = "<s> {0} </s>".format(eachPoem)
            splittedPoem = eachPoem.split(" ")
            for word in splittedPoem:
                totalList.append(word.lower())
            bigramArray.append(languagemodel(eachPoem, 2))
        uniquePoem = list(set(totalList))

readJSON()

uniqueSIZE = len(uniquePoem)

#creates one hot vectors
def createVectors():
    for i in range(uniqueSIZE):
        tempVector = [0 if x !=i else 1 for x in range(uniqueSIZE)]
        oneHotVectorArray.append(tempVector)
        oneHotVectorDict.update({uniquePoem[i] : i})

createVectors()


def newWord():
    randNum = random.randint(0, len(uniquePoem))
    return randNum

def countBigram():
    global totalBigram
    for sent in bigramArray:
        for pair in sent:
            totalBigram += 1

countBigram()

model = dy.Model()
pW = model.add_parameters((10, len(uniquePoem)))
pb = model.add_parameters(10)
pU = model.add_parameters((len(uniquePoem), 10))
pd = model.add_parameters(len(uniquePoem))

trainer = dy.SimpleSGDTrainer(model)


EPOCHS = 10


def getDy():
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for data in bigramArray:
            for pair in data:
                dy.renew_cg()
                splittedPair = pair.split(" ")
                #x is input, y is output
                x = dy.inputVector(oneHotVectorArray[oneHotVectorDict.get(splittedPair[0])])
                y = oneHotVectorDict.get(splittedPair[1])

                yHat = pU * dy.tanh(pW * x + pb) + pd

                loss = dy.pickneglogsoftmax(yHat, y)

                epoch_loss += (loss.scalar_value() / totalBigram)
                loss.backward()
                trainer.update()

        print("Epoch %d. loss = %f" % (epoch, epoch_loss))

getDy()


def findNewline(poem):
    count = 0
    for word in poem:
        if word == "/n":
            count += 1
    return count

lineNumber = int(input("How many lines of poems do you want to be created?: "))

#sometimes the problem of infinite loop, please run again. I'm sorry for that.
def startPredict(start):
    print("Generating is started!")
    totalProb = 0
    newLineCounter=0
    i=0
    poemCount=1
    starter = start
    totalPoem = starter
    siir = []
    while(newLineCounter < lineNumber):
        i+=1
        x=dy.inputVector(oneHotVectorArray[oneHotVectorDict.get(starter)])
        yHat = pU * dy.tanh(pW * x + pb) + pd
        ind = np.argmax(yHat.npvalue())
        prob = math.log(list(yHat.npvalue())[ind])

        kelime = uniquePoem[ind]

        poemLine = str(totalPoem).split(" ")
        newlineNumber = findNewline(poemLine)
        lastWord = poemLine[-1]
        if newlineNumber == lineNumber-1:
            break
        if kelime=="\n" and len(poemLine)<4:
            kelime = uniquePoem[newWord()]
        elif kelime == "\n" and lastWord != "\n":
            totalPoem = totalPoem + " /n "
            newLineCounter+=1
            kelime = uniquePoem[newWord()]
        elif kelime == "</s>":
            print("</s> is found!")
            i=2
            totalProb=1
            break
        elif kelime == "<s>":
            kelime = uniquePoem[newWord()]
        totalPoem = totalPoem + " " + kelime
        starter = kelime
        totalProb += prob

    print(totalPoem)
    # print("{0}. Poem : ".format(i) + totalPoem)
    perplexity = totalProb / (i - 1)
    # print("{0}. Perplexity ".format(i) + str(perplexity))
    print("Perplexity : " + str(perplexity) )

#The following code fragment was written into the loop to create 5 times the poem.
# However, no run was observed in that case.
# The report contains a personal comment section.
startPredict(uniquePoem[newWord()])
