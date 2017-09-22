from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from surprise import AlgoBase
from surprise import Dataset
from surprise import evaluate

class SVD_A(AlgoBase):
    #our initializer (constructor)
    def __init__(self,n_factors = 35,n_iter = 40,init_mean=0,
                 init_std_dev=.1, learnr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 verbose=False):

        self.n_factors = n_factors
        self.n_iter = n_iter
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev

        self.lr_bu = lr_bu if lr_bu is not None else learnr_all
        self.lr_bi = lr_bi if lr_bi is not None else learnr_all
        self.lr_pu = lr_pu if lr_pu is not None else learnr_all
        self.lr_qi = lr_qi if lr_qi is not None else learnr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.__v = verbose

        AlgoBase.__init__(self)

    #used for training the machine, if not specified in surprise it uses the default way, but we need our own specification
    def train(self, trainset):
        AlgoBase.train(self, trainset)
        #we call our written sgd function here
        self.sgd(trainset)

    #the main part of our SVD learning(training)
    def sgd(self, trainset):
        #we first initialize the parameters we use in sgd
        mu = self.trainset.global_mean
        bias_u = np.zeros(trainset.n_users,np.double)
        bias_i = np.zeros(trainset.n_items, np.double)
        Pu = np.sqrt(5.0/self.n_factors) * np.random.random_sample((trainset.n_users,self.n_factors))
        Qi = np.sqrt(5.0/self.n_factors) * np.random.random_sample((trainset.n_items,self.n_factors))

        for iter in range(self.n_iter):
            for u,i,r in trainset.all_ratings():
                prediction = mu + bias_u[u] + bias_i[i]
                prediction += Pu[u, :].dot(Qi[i, :].T)
                err = r - prediction

                # Update biases
                bias_u[u] += self.lr_bu * \
                                    (err - self.reg_bu * bias_u[u])
                bias_i[i] += self.lr_bi * \
                                    (err - self.reg_bi * bias_i[i])

                #Update latent factors
                Pu[u, :] += self.lr_pu * \
                                        (err * Qi[i, :] - \
                                         self.reg_pu * Pu[u,:])
                Qi[i, :] += self.lr_qi * \
                                        (err * Pu[u, :] - \
                                         self.reg_qi * Qi[i,:])

        #now after learning everything we make the learned parameters golbal to be used by our estimate function
        self.bu = bias_u
        self.bi = bias_i
        self.Pu = Pu
        self.Qi = Qi

    #after training our model now we can actually specify what the model is supposed to do in estimate
    #estimate function is ai main component in any custom surprise algorithm
    #all we do in this estimate is updating our biases as we see known users and the rest would be taken care of
    def estimate(self, u, i):

        estimate = self.trainset.global_mean

        if self.trainset.knows_user(u):
            estimate += self.bu[u]

        if self.trainset.knows_item(i):
            estimate += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            estimate += np.dot(self.Qi[i], self.Pu[u])


        return estimate
class SVDplusplus_A(AlgoBase):
    #our initializer (constructor)
    def __init__(self,n_factors = 15,n_iter = 40,init_mean=0,
                 init_std_dev=.1, learnr_all=.008,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,lr_yj = None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, reg_yj = None,
                 verbose=False):

        self.n_factors = n_factors
        self.n_iter = n_iter
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev


        #we have added a regularization and learning rate for Y matrix, can be initialized by user if needed
        self.lr_bu = lr_bu if lr_bu is not None else learnr_all
        self.lr_bi = lr_bi if lr_bi is not None else learnr_all
        self.lr_pu = lr_pu if lr_pu is not None else learnr_all
        self.lr_qi = lr_qi if lr_qi is not None else learnr_all
        self.lr_yj = lr_yj if lr_yj is not None else learnr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all
        self.__v = verbose

        AlgoBase.__init__(self)

    #used for training the machine, if not specified in surprise it uses the default way, but we need our own specification
    def train(self, trainset):
        AlgoBase.train(self, trainset)
        #we call our written sgd function here
        self.sgd(trainset)

    #As discussed in many papers and atricles the actual optimized learning method for SVD++ is ALS
    #Although as the authors of SVD++ found and as we use it here, SGD being used in a little bit different way than normal
    #can be fast and even faster than ALS
    #the sgd method here as discussed is a little bit different than the one in SVD class

    def sgd(self, trainset):

        #we first initialize the parameters we use in sgd
        mu = self.trainset.global_mean
        bias_u = np.zeros(trainset.n_users,np.double)
        bias_i = np.zeros(trainset.n_items, np.double)
        Pu = np.sqrt(5.0/self.n_factors) * np.random.random_sample((trainset.n_users,self.n_factors))
        Qi = np.sqrt(5.0/self.n_factors) * np.random.random_sample((trainset.n_items,self.n_factors))
        #Matrix for measuring user feed back
        #this matrix is different than the factor matrices and would be more efficient when  initalized totally in random
        #with no special bound
        Yj = np.random.normal(self.init_mean,self.init_std_dev,(trainset.n_items,self.n_factors))
        #the matrix for the user implicit feedback as rows and number of factors as columns
        user_implct_feedbk = np.zeros(self.n_factors,np.double)

        for iter in range(self.n_iter):
            for u,i,r in trainset.all_ratings():
                #we should note that because of parsing through all actual ratings
                # The SVD++ is more costly than SVD itself one thing we can do optimize SVD++
                #is to compute the dot product the two matrices by hand
                #after all matrix multiplications (even dot products) are one of the costliest
                #when dealing matrices

                #the matrix is nothing but all items having been rated by user u
                #this should be repeated for all users (that's one the costly part of SVD++)
                items_u = [j for (j, _) in trainset.ur[u]]

                sqrt_Iu = np.sqrt(len(items_u))

                # compute user implicit feedback (the actual one)
                user_implct_feedbk = np.zeros(self.n_factors, np.double)

                for j in items_u:
                    for f in range(self.n_factors):
                        user_implct_feedbk[f] += Yj[j, f] / sqrt_Iu

                # Compute error and dot product by hand (as discussed)

                dot = 0
                for f in range(self.n_factors):
                    #based on the formula of SVD++
                    dot += Qi[i, f] * (Pu[u, f] + user_implct_feedbk[f])

                #like before and based on formula
                err = r - (mu + bias_u[u] + bias_i[i] + dot)

                # update biases with regularization parameter
                bias_u[u] += self.lr_bu * (err - self.reg_bu * bias_u[u])
                bias_i[i] += self.lr_bi * (err - self.reg_bi * bias_i[i])

                # update factors with regularization parameters
                #we agin here tried to be optimized so instead of parsing an array inside of it we do it iteratively
                #inside a loop
                for f in range(self.n_factors):
                    puf = Pu[u, f]
                    qif = Qi[i, f]

                    #updating the actual factor matrices
                    Pu[u, f] += self.lr_pu * (err * qif - self.reg_pu * puf)
                    Qi[i, f] += self.lr_qi * (err * (puf + user_implct_feedbk[f]) -
                                         self.reg_qi * qif)

                    #the update of values in Yj should be in in an inner loop for all items rated by user u
                    for j in items_u:
                        Yj[j, f] += self.lr_yj * (err * qif / sqrt_Iu -
                                             self.reg_yj * Yj[j, f])

        #now after learning everything we make the learned parameters golbal to be used by our estimate function
        self.bu = bias_u
        self.bi = bias_i
        self.Pu = Pu
        self.Qi = Qi
        self.Yj = Yj

    #after training our model now we can actually specify what the model is supposed to do in estimate
    #estimate function is ai main component in any custom surprise algorithm
    #all we do in this estimate is updating our biases as we see known users and the rest would be taken care of
    def estimate(self, u, i):

        estimate = self.trainset.global_mean

        if self.trainset.knows_user(u):
            estimate += self.bu[u]

        if self.trainset.knows_item(i):
            estimate += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            #number of items rated by user
            num_I = len(self.trainset.ur[u])

            #updating user implict feedback matrix based on what we got
            user_implct_feedbk = (sum(self.Yj[j] for (j, _)
                               in self.trainset.ur[u]) / np.sqrt(num_I))
            estimate += np.dot(self.Qi[i], self.Pu[u] + user_implct_feedbk)


        return estimate
