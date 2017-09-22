from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from surprise import AlgoBase
from surprise import Dataset
from surprise import evaluate


#the algorithm is very similar to SVD++ (in fact SVD++ is a specialization of this algorithm) note that we have no longer pu
#parameter as we have no user factor matrix, we instead have a matrix X (in formula shown as R or N) which handles all
#the explicit ratings given by each user
class AFM(AlgoBase):
    def __init__(self,n_factors = 20,n_iter = 40,init_mean=0,
                 init_std_dev=.1, learnr_all=.009,
                 reg_all=.04, lr_bu=None, lr_bi=None, lr_qi=None,lr_yj = None, lr_xu = None,
                 reg_bu=None, reg_bi=None, reg_qi=None, reg_yj = None, reg_xu = None
                 verbose=False):

    self.n_factors = n_factors
    self.n_iter = n_iter
    self.init_mean = init_mean
    self.init_std_dev = init_std_dev


    #we have added a regularization and learning rate for Y matrix, can be initialized by user if needed
    #we also have added one more parameter X (for explicit rating which would be update in sgd iterations)
    self.lr_bi = lr_bi if lr_bi is not None else learnr_all
    self.lr_qi = lr_qi if lr_qi is not None else learnr_all
    self.lr_yj = lr_yj if lr_yj is not None else learnr_all
    self.lr_xu = lr_xu if lr_xu is not None else learnr_all
    self.reg_bi = reg_bi if reg_bi is not None else reg_all
    self.reg_qi = reg_qi if reg_qi is not None else reg_all
    self.reg_yj = reg_yj if reg_yj is not None else reg_all
    self.reg_xu = reg_xu if reg_xu is not None else reg_all
    self.__v = verbose

    def train(self, trainset):
        AlgoBase.train(self, trainset)
        #we call our written sgd function here
        self.sgd(trainset)

    #The asymmetric factor model deals with only a dot product of two item factor matrix row (vecotr)
    #There some parts similar to SVD++ but since there is no more user involved the big for loop and estimate functions should be
    #coded totally diffrently
    def sgd(self, trainset):
        mu = self.trainset.global_mean
        bias_i = np.zeros(trainset.n_items, np.double)
        Qi = np.sqrt(5.0/self.n_factors) * np.random.random_sample((trainset.n_items,self.n_factors))

        Yj = np.random.normal(self.init_mean,self.init_std_dev,(trainset.n_items,self.n_factors))

        user_implct_feedbk = np.zeros(self.n_factors,np.double)
    def estimte (self, i1, i2):
        if self.trainset.knows_item(i1):
            estimate += self.bi[i1]
        if self.trainset.knows_item(i2):
            estimate += self.bi[i2]
