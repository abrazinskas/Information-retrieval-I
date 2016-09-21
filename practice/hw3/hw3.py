# this file contains some functions that were used to obtains results.
from RankNet import  RankNet
from PointWiseNN import PointWiseNN
from LambdaRank import LambdaRank
from RankNetMod import RankNetMod
from query import load_queries
from support import CrossFold, NDCG,getRankedList,plotBars
import numpy as np
from timeit import default_timer as timer


# do cross validaiton
def doCV():
     pointWise=PointWiseNN(64)
     rankNet=RankNet(64)
     rankNetMod=RankNetMod(64)
     lambdaRank=LambdaRank(64)


     cf=CrossFold([pointWise,rankNet,rankNetMod,lambdaRank])
     res=cf.run('data/',1)
     plotBars(res[0],x_labels=["pointWise",'rankNet','rankNetMod','lambdaRank'],y_label="mNDCG",title="Cross Validation results")
     plotBars(res[1],x_labels=["pointWise",'rankNet','rankNetMod','lambdaRank'],y_label="seconds",title="Runtime")
    # print(res)

doCV()



# run this method to compare the origina rankNet and the modified version
def speedTestAndEval():
    rankNet=RankNet(64)
    rankNetMod=RankNetMod(64)

    file='data/Fold2/train.txt'
    queries=load_queries(file,64)
    start=timer()
    rankNet.train_with_queries(queries,4)
    print("- Took %.2f sec to train rankNet " % (timer() - start))

    start=timer()
    rankNetMod.train_with_queries(queries,4)
    print("- Took %.2f sec to train rankNetMod " % (timer() - start))


    print('----Now lets evaluate--------')
    ndcg=NDCG(1)
    testQueries=load_queries('data/Fold2/test.txt',64)
    for name,ranker in zip(['rankNet','rankNetMod'],[rankNet,rankNetMod]):
        r=[]
        for q in testQueries:
            rel=getRankedList(ranker,q)
            r.append(ndcg.run(rel,max_c=np.sum(rel)))
        print('mNDCG for '+name+": ")
        print(np.mean(r))


#speedTestAndEval()



