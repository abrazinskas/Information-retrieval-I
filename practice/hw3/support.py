import numpy as np
import glob
import os
from query import load_queries
import copy
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['legend.loc'] = 'best'
from timeit import default_timer as timer

# notice that this is a modified version of NDCG with relative normalization
class NDCG():
    __cache = {}
    __max_rel = None

    def __init__(self, max_rel):
        self.__max_rel = max_rel

    # dcs: documents collections returned by the queries
    # assuming that each document is represented by it's relevance!
    # e.g. [1,4,3,2] <- collection of 4 documents
    # max_c:maximum number of ones in our case
    def run(self, dc,max_c):
        k = len(dc)
        max_c=int(max_c)# making sure that max_c is int
        if(max_c==0):
            return 0
        Z = self.__getNorm(k,max_c)
        res = 0
        for r in range(k):
            res += self.__score(dc[r], r + 1)
        return Z * res

    # computes ndcg on multiple collections of documents
    # returns an array of values
    def runOnCol(self, dcs):
        n = len(dcs)
        res = np.zeros(n)
        for i in range(n):
            res[i] = self.run(dcs[i])
        return res


    # k is the length
    # c: max number of maximum relevances
    def __getNorm(self, k,max_c):
        if (k in self.__cache and max_c in self.__cache[k]):
            return self.__cache[k][max_c]
        Z = 0
        for r in range(1, max_c + 1):
            Z += self.__score(self.__max_rel, r)
        Z = 1 / Z
        # storing in cache
        self.__cache[k]={max_c:Z}
        return Z

    def __score(self, rel, rank):
        return (math.pow(2, rel) - 1) / math.log(1 + rank, 2)

# we score documents in the collection
# then sort by the score
# and return back the actual relevance list
def getRankedList(model,query,justRel=True):
    scores=model.score(query)
    labels=query.get_labels()
    oIds=range(len(labels))
    if(justRel==True):
        return [y for (x,y) in sorted(zip(scores,labels),reverse=True)]
    else:
        ids=[]
        rel=[]
        for (score,lab,id) in sorted(zip(scores,labels,oIds),reverse=True,key=lambda x: x[0]):
            ids.append(id)
            rel.append(lab)
        return (ids,rel)

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + np.exp(-z))


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


class CrossFold():
    models= None # a list of models
    ndcg=None # for evaluation

    # models: a list of models
    def __init__(self,models):
        self.models=models
        self.ndcg=NDCG(1)


    def run(self,mainFolder,epochs):
        folders = get_immediate_subdirectories(mainFolder)
        #folders = random.sample(get_immediate_subdirectories(mainFolder),2)
        ndcgs=[[] for i in range(len(self.models))]
        elapsed=np.zeros(len(self.models)) # for timing

        for i,folder in enumerate(folders):
            print("fold "+str(i+1))

            trainFile=mainFolder+folder+"/train.txt"
            train_queries=load_queries(trainFile,64)
            testFile=mainFolder+folder+"/test.txt"
            testQueries=load_queries(testFile,64)

            for i,model in enumerate(self.models):
                now=timer()
                model.train_with_queries(train_queries,epochs)
                elapsed[i]+=timer()-now
                # evaluation
                ndcgs[i]+=self.__evalaute(testQueries,model)
        return ([np.mean(n) for n in ndcgs],elapsed)


    # returns a tuple of list of ndcgs
    def __evalaute(self,queries,model):
        ndcgs=[]
        for q in queries:
            l=getRankedList(model,q)[:10]
            ndcgs.append(self.ndcg.run(l,max_c=np.sum(l)))
        return ndcgs

    # just to test models
    def OneFoldTest(self,folder,model,epochs):
        testFile=folder+"/test.txt"
        testQueries=load_queries(testFile,64)


        trainFile=folder+"/train.txt"
        trainQueries=load_queries(trainFile,64)


        model.train_with_queries(trainQueries,epochs)


        ndcgs=[]
        for q in testQueries:
            ndcgs.append(self.ndcg.run(getRankedList(model,q)))


        return (np.mean(ndcgs))



def plotBars(bars,x_labels,y_label,title):

    ind = np.arange(len(bars))
    width = 0.2
    fig, ax = plt.subplots()
    rec1=ax.bar(ind, bars, width, color='r',align='center')

    ax.set_xticks(ind)
    plt.ylabel(y_label)
    ax.set_xticklabels(x_labels)
    fig.suptitle(title.lower().replace(" ","_"), fontsize=20)
    plt.grid(True)
    plt.legend()


    plt.savefig(title.lower().replace(" ","_") + '.jpg')






