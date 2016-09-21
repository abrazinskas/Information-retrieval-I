# this is the modified version of RankNet.
# the intuition behind the model is given the analysis.pdf file
__author__ = 'arthur'

import itertools
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from itertools import count
import query
import math
import copy

from support import sigmoid,NDCG, getRankedList

NUM_EPOCHS = 500

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 500
LEARNING_RATE = 0.025
MOMENTUM = 0.95


THEANO_FLAGS="floatX=float32"

# TOD: Implement the lambda loss function
def lambda_loss(output, lambdas):
    return -output*lambdas


class RankNetMod:

    NUM_INSTANCES = count()

    def __init__(self, feature_count):
        self.feature_count = feature_count
        self.output_layer = self.build_model(feature_count,1,BATCH_SIZE)
        self.iter_funcs = self.create_functions(self.output_layer)
        self.ndcg=NDCG(1)

    # train_queries are what load_queries returns - implemented in query.py
    def train_with_queries(self, train_queries, num_epochs):

        try:
            now = time.time()
            for epoch in self.train(train_queries):
                if epoch['number'] % 5 == 0:
                    print("Epoch {} of {} took {:.3f}s".format(
                    epoch['number'], num_epochs, time.time() - now))
                    print("training loss:\t\t{:.6f}\n".format(epoch['train_loss']))
                    now = time.time()
                if epoch['number'] >= num_epochs:
                    break
        except KeyboardInterrupt:
            pass

    def score(self, query):
        feature_vectors = query.get_feature_vectors()
        scores = self.iter_funcs['out'](feature_vectors)
        return scores


    def build_model(self,input_dim, output_dim,
                    batch_size=BATCH_SIZE):
        """Create a symbolic representation of a neural network with `intput_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """
        print "input_dim",input_dim, "output_dim",output_dim
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, input_dim),
        )

        l_hidden = lasagne.layers.DenseLayer(
            l_in,
            num_units=200,
            nonlinearity=lasagne.nonlinearities.tanh,
        )


        l_out = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        return l_out

    # Create functions to be used by Theano for scoring and training
    def create_functions(self, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM, L1_reg=0.0000005, L2_reg=0.000003):
        """Create functions for training, validation and testing to iterate one
           epoch.
        """
        X_batch = X_tensor_type('x')
        y_batch = T.fvector('y')

        output_row = lasagne.layers.get_output(output_layer, X_batch, dtype="float32")
        output = output_row.T

        output_row_det = lasagne.layers.get_output(output_layer, X_batch,deterministic=True, dtype="float32")

        # TOD: Change loss function
        # Point-wise loss function (squared error) - comment it out
        # loss_train = lasagne.objectives.squared_error(output,y_batch)
        # Pairwise loss function - comment it in
        loss_train = lambda_loss(output, y_batch)

        loss_train = loss_train.mean()

        # TODO: (Optionally) You can add regularization if you want - for those interested
        L1_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l1)
        L2_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
        loss_train = loss_train.mean() + L1_loss * L1_reg + L2_loss * L2_reg

        # Parameters you want to update
        all_params = lasagne.layers.get_all_params(output_layer)

        # Update parameters, adam is a particular "flavor" of Gradient Descent
        updates = lasagne.updates.adam(loss_train, all_params,learning_rate=learning_rate)


        # Create two functions:

        # (1) Scoring function, deterministic, does not update parameters, outputs scores
        score_func = theano.function(
            [X_batch], output_row_det,
        )

        # (2) Training function, updates the parameters, outpust loss
        train_func = theano.function(
            [X_batch,y_batch], loss_train,
            updates=updates,
            # givens={
            #     X_batch: dataset['X_train'][batch_slice],
            #     # y_batch: dataset['y_valid'][batch_slice],
            # },
        allow_input_downcast=True)

        print "finished create_iter_functions"
        return dict(
            train=train_func,
            out=score_func,
        )


    # I assume that this one has to return a list of \lambda_i, i.e. a lambda for each document
    def lambda_function(self, didslabels, scores):
        # at this stage I could avoid storing S collection
        #
        S,I=self.__get_S_and_I(didslabels[1])

        n=len(didslabels[1])
        lamb=np.zeros(n,dtype='float32')
        for i in range(n):
            res=0 # left hand side lambda in \lambda_{i} equation
            if(i in I):
                didslabels[1][i],didslabels[1][i+1]=didslabels[1][i+1],didslabels[1][i] # swapping labels
                didslabels[1][i+1],didslabels[1][i]=didslabels[1][i],didslabels[1][i+1] # swapping labels back

                s=(scores[i],scores[i+1])
                res+=self.__compute_lambda(s,S[i])
            if(i!=0 and i-1 in I):
                didslabels[1][i-1],didslabels[1][i]=didslabels[1][i],didslabels[1][i-1] # swapping
                didslabels[1][i],didslabels[1][i-1]=didslabels[1][i-1],didslabels[1][i] # swapping back

                s=(scores[i-1],scores[i])
                res-=self.__compute_lambda(s,S[i-1])
            lamb[didslabels[0][i]]=res
        return lamb



    # scores for i,j tuple
    # S_{i,j} value (1,-1, or 0)
    def __compute_lambda(self,scores,S):
        sig=sigmoid(scores[0]-scores[1])
        res=((1-S)/2) - 1/(1+math.exp(sig))
        return sigmoid(res)

    # returns :
    #           1. a list of -1, 0, or 1 depending of the label sequence
    #           2. a dict. for I where i appears on the left side
    # we don't need I where i appears on the right because we have binary relevance.
    def __get_S_and_I(self,labels):
        I={}
        S=[]
        n=len(labels)
        if(n<2):
            return []
        for i in range(0,n-1):
            if(labels[i]>labels[i+1]):
                val=1
                I[i]=i+1
            if(labels[i]<labels[i+1]):
                val=-1
            if(labels[i]==labels[i+1]):
                val=0
            S.append(val)
        return S,I


    def compute_lambdas_theano(self,query, labels):
        scores = self.score(query).flatten()

        # need to sort scores accordingly
        s= [scores[id] for id in labels[0]]

        result = self.lambda_function(labels,s)
        return result

    def train_once(self, X_train, query, docIdsLabels):

        # TOD: Comment out to obtain the lambdas
        lambdas = self.compute_lambdas_theano(query,docIdsLabels)
        #lambdas.resize((BATCH_SIZE, ))

        # Otherwise it breaks the whole code
        #X_train.resize((BATCH_SIZE, self.feature_count),refcheck=False)

        # TOD: Comment out (and comment in) to replace labels by lambdas
        batch_train_loss = self.iter_funcs['train'](X_train, lambdas)

        # this functions seems to do the actual training
        # batch_train_loss = self.iter_funcs['train'](X_train, labels)
        return batch_train_loss

    def train(self, train_queries):
        print('training RankNetMod')
        X_trains = train_queries.get_feature_vectors()

        queries = train_queries.values()

        for epoch in itertools.count(1):
            batch_train_losses = []
            random_batch = np.arange(len(queries))
            np.random.shuffle(random_batch)
            for index in xrange(len(queries)):
                random_index = random_batch[index]
                # I assume that this is a wrong method because labels vector will be one-hot-vector
                # docLabels = queries[random_index].get_labels()
                docIdsLabels = getRankedList(self,queries[random_index],justRel=False) # ids:label pairs collection

                # stochastic training I suppose
                batch_train_loss = self.train_once(X_trains[random_index],queries[random_index],docIdsLabels)
                batch_train_losses.append(batch_train_loss)


            avg_train_loss = np.mean(batch_train_losses)

            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
            }

