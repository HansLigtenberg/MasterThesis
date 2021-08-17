"""
Test7.py

Purpose: Working towards the first empirical analysis

Version:
    1   Prepare data
    2   Create list of hyper parameters
    3   Add training fraction to hyper parameters
        Create function to unpack hyper paramters and start estimation
        Parallel implementation using Pathos
    4   Store messages in list of dicts
    5   Change function names
    6   Repair messages list
    7   Add messages to all returned functions and set constant to just a number.
        No momentum replaced by zero
    8   Max tasks per child is set to 1. Hopefully more stable
    9   Try to use pebble such that a time out can be set -> does not work due to pickle problem
    10  Switch back to pathos
    11  Do not standardise factor and return
    12  Initialise network with bias in loss node. Needs Modules34 or higher
    13  Switch form random search to grid search
    14  Check for models already estimated and do not reestimate those
    15  Reduce number of models to be estimated and set maximum on number of updates
    16  Handles error better, such that last models can be trained as well
    17  Reestimate smaller number of models and improve estimation procedure
        And remove stochastic part
    18  Estimate sigma instead of log sigma
    19  Fix the switch of HML and M, back to estimating log sigma

Author: Hans Ligtenberg
"""

import numpy as np
import Modules as mod
import matplotlib.pyplot as plt
import ReadData2
import multiprocessing as mp
import pathos
import pickle
import dill
import pandas as pd
import os
import logging

def prepareData(growth=True, standardiseMaxMin=False, a=-1, b=1, filterVars=None, filterValues=None):
    """Loads the data, select the correct columns and normalises it to have 
    mean zero and std one. When growth is True take the log-growth of INF and IP.
    By default standardise to have zero mean and unit standardeviation. Other
    option is to standardise such that all values lie between a and b."""
    df=ReadData2.loadData()
    if filterVars is not None:
        df=df.drop(df[(df[filterVars]>filterValues).any(axis=1)].index)
    if growth:
        df['INF']=np.log(df['INF'])-np.log(df['INF'].shift(1))
        df['IP']=np.log(df['IP'])-np.log(df['IP'].shift(1))
    df[['HML','M']]=df[['HML','M']].shift(-1)
    df=df.drop(['VP 10', 'VP 20', 'VP 30', 'RF'], axis=1)
    df=df.dropna()
    df1=df[['HML', 'M']]
    df2=df.drop(['HML', 'M'], axis=1)
    if standardiseMaxMin:
        df2=a+(df2-df2.min())*(b-a)/(df2.max()-df2.min())
    else:
        for col in df2.columns:
            df2[col]=(df2[col]-df2[col].mean())/df2[col].std()
    return pd.merge(df1,df2, how='outer',left_index=True, right_index=True)

def splitDataToInput(df, fracTrain, outputNodeNr, factor='M', R='HML', X=['DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP']):
    """Splits df in train and validation data where fracTrain fraction is
    used as training data and the remaining for validation data. Also constructs
    lists of inputs in correct format to be passed into optimisers."""
    x=df[X].to_numpy()
    f=df[factor].to_numpy().reshape((-1,1))
    R=df[R].to_numpy().reshape((-1,1))
    if fracTrain==1:
        inputX=[[0,x]]
        addInput=[[outputNodeNr,[['factor', f],['R', R]]]]
        return [inputX, addInput], None
    else:
        train=int(df.shape[0]*fracTrain)
        inputXTrain=[[0,x[:train,:]]]
        addInputTrain=[[outputNodeNr,[['factor', f[:train,:]],['R', R[:train,:]]]]]
        inputXValid=[[0,x[train:,:]]]
        addInputValid=[[outputNodeNr,[['factor', f[train:,:]],['R', R[train:,:]]]]]
        return [inputXTrain, addInputTrain], [inputXValid, addInputValid]

def lrNList():
    """Creates a list of functions to return learning rates. Also returns a description
    about which function is used."""
    params=[[0.01,0.001,250],[0.01,0.001,1000],[0.001,0.0001,250],[0.001,0.0001,100]]
    l=[]
    msg=[]
    for eps0, epsTau, tau in params:
        def lr(k):
            return (1-k/tau)*eps0+(k/tau)*epsTau if k<tau else epsTau
        l.append(lr)
        msg.append({'lr':f'Lin({eps0},{epsTau},{tau})'})
    return l, msg

def lrRMSPropList():
    """Creates a list of functions to return learning rates. Also returns a description
    about which function is used."""
    l=[lambda x: 0.0001, lambda x: 0.001, lambda x: 0.01]
    msg=[{'lr':0.0001},{'lr':0.001},{'lr':0.01}]
    for i in range(len(l)):
        l[i].__name__='lr'
    return l, msg

def aList():
    """Creates a list of functions to return momentum parameters. Also returns
    a description about which function is used."""
    l=[lambda x: 0.5]
    for i in range(len(l)):
        l[i].__name__='a'
    msg=[{'a':0.5}]
    return l, msg

def rhoList():
    """Creates a list of functions to return gradient acc parameters. Also returns
    a description about which function is used."""
    l=[lambda x: 0.5]
    for i in range(len(l)):
        l[i].__name__='rho'
    msg=[{'rho':0.5}]
    return l, msg

def p(rng, msg):
    """Selects patience for early stopping. For now constant."""
    return 200

def step(rng, msg):
    """Selects step size for early stopping. For now constant."""
    return 1

def fracTrain(rng, msg):
    """Selects training fraction. For now constant."""
    return 0.8

def seed(rng, msg):
    """Selects training fraction. For now constant."""
    return 19980528

def seedGen(rng=None):
    """Randomly draws a seed."""
    if rng is None:
        rng=np.random.default_rng()
    return rng.integers(100000)
    
def hyperParametersDeterministic(rng=None, draws=5, growth=True, standardiseMaxMin=False, filterVars=None, filterValues=None):
    """Constructs a list of hyper parameters. Parameters follow a grid. Draws
    is the number each specification is repeated, but with a different seed."""
    df=prepareData(growth=growth, standardiseMaxMin=standardiseMaxMin, filterVars=None, filterValues=None)
    res=[]
    msgs=[]
    architectures=[[9,9],[9,9,9],[9,9,9,9],[9,9,9,9,9,9]]
    methods=['Nesterov', 'RMSProp']
    lrN, msgLrN=lrNList()
    lrRMSProp, msgLrRMSProp=lrRMSPropList()
    a, msgA=aList()
    rho, msgRho=rhoList()
    B=[128]
    p=200
    step=1
    run=0
    maxUpdates=np.inf
    fracTrain=0.8
    for draw in range(draws):
        for arch in architectures:
            outputNodeNr=len(arch)
            trainData, validData=splitDataToInput(df, fracTrain, outputNodeNr)
            for method in methods:
                for momentum, momentumMsg in zip(a, msgA):
                    for b in B:
                        if method=='Nesterov':
                            for lr, lrMsg in zip(lrN, msgLrN):
                                res.append([arch, method, lr, momentum, b, p, step,
                                            seedGen(rng), maxUpdates, trainData, validData, run])
                                msgs.append({**momentumMsg, **lrMsg})
                                run+=1
                        elif method=='RMSProp':
                            for lr, lrMsg in zip(lrRMSProp, msgLrRMSProp):
                                for r, rMsg in zip(rho, msgRho):
                                    res.append([arch, method, lr, momentum, b, p, step,
                                            seedGen(rng), maxUpdates, r, trainData, validData, run])
                                    msgs.append({**momentumMsg, **lrMsg, **rMsg})
                                    run+=1
                        else: raise Exception('Method not recognised.')                    
    return res, msgs

def OLS(x, y):
    """Performs OLS of 1~x on y, samples in rows. Returns
    estimated beta and sigma squared."""
    X=np.hstack((np.full((len(x),1),1), x))
    n,k=x.shape
    beta=np.linalg.inv(X.T@X)@X.T@y
    eps=y-X@beta
    sigma2=eps.T@eps/(n-k)
    return beta, sigma2

def train(l, locLog, locResults):
    import Modules as mod
    process=pathos.core.getpid()
    fh=logging.FileHandler(locLog, mode='a')
    fh.setLevel(logging.DEBUG)
    fileFormat=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fileFormat)
    logger=pathos.logger(level=logging.DEBUG, handler=fh)
    logger.addHandler(fh)
    
    if not os.path.isfile(locResults+f'k/k_{l[-1]}'):   
        logger.info(f'Process {process} started run {l[-1]}')
    
        arch=l[0]
        method=l[1]
        if method=='Nesterov':
            lr, a, B, p, step, seed, maxUpdates, trainData, validData, run=l[2:]
            factor=trainData[1][0][1][0][1]
            R=trainData[1][0][1][1][1]
            (alpha, beta), sigma2=OLS(factor, R)
            lossBias=np.array([alpha.reshape(1), beta.reshape(1), np.log(np.sqrt(sigma2.reshape(1)))])
            rng=np.random.default_rng(seed)
            network=mod.FeedForwardLossLogSigma(arch,lossBias,rng=rng,ReLUBias=0.01)
            losses, k, p, lossK=network.trainNesterov(trainData, lr, a, p, step, len(arch), B, 'output', validData, maxUpdates)
        elif method=='RMSProp':
            lr, a, B, p, step, seed, maxUpdates, rho, trainData, validData, run=l[2:]
            factor=trainData[1][0][1][0][1]
            R=trainData[1][0][1][1][1]
            (alpha, beta), sigma2=OLS(factor, R)
            lossBias=np.array([alpha.reshape(1), beta.reshape(1), np.log(np.sqrt(sigma2.reshape(1)))])
            rng=np.random.default_rng(seed)
            network=mod.FeedForwardLossLogSigma(arch,lossBias,rng=rng,ReLUBias=0.01)
            losses, k, p, lossK=network.trainRMSProp(trainData, lr, a, rho, p, step, len(arch), B, 'output', validData, maxUpdates)
        else: 
            raise Exception('Method not recognised.')
        finalParams=network.storeParameters()
        finalLoss=network.batchLoss(validData[0], len(arch), 'output', validData[1])
        
        with open(locResults+'Losses/Losses_'+str(run), "wb") as file:   
            pickle.dump(losses, file)
        with open(locResults+'k/k_'+str(run), "wb") as file:   
            pickle.dump(k, file)
        with open(locResults+'p/p_'+str(run), "wb") as file:   
            pickle.dump(p, file)
        with open(locResults+'Final_parameters/Final_parameters_'+str(run), "wb") as file:   
            pickle.dump(finalParams, file)
        with open(locResults+'Final_loss/Final_loss_'+str(run), "wb") as file:   
            pickle.dump(finalLoss, file)
        with open(locResults+'Loss_k/Loss_k_'+str(run), "wb") as file:   
            pickle.dump(lossK, file)    
        logger.info(f'Process {process} finished run {l[-1]}')
    else:
        logger.info(f'Run {l[-1]} already estimated')
    
    logger.removeHandler(fh)
    return 
          

def main():
    #Magic numbers
    seed=19980528   
    locResults=os.getcwd()+'/Results/Estimates/'
    locLog=os.getcwd()+'/Log.log'

    #Initialisations
    randomRng=np.random.default_rng()
    rng=np.random.default_rng(seed)
    params, msgs=hyperParametersDeterministic(rng, draws=5, growth=True, standardiseMaxMin=False, filterVars=None, filterValues=None)
    with open(locResults+'Hyper-Parameters', "wb") as file1, open(locResults+'Messages', "wb") as file2:
        dill.dump(params, file1)
        dill.dump(msgs, file2)
    print(f'{len(params)} models will be estimated')
        
    #Shuffle params to minimise training the same models as the other computer
    randomRng.shuffle(params)
    
    #Start training
    pool=pathos.multiprocessing.ProcessingPool(maxtasksperchild=1)
    res=pool.map(train, params, len(params)*[locLog], len(params)*[locResults])
    pool.close()
    pool.join()
    pool.clear()
    
if __name__=="__main__":
    __spec__ = None
    pathos.helpers.freeze_support()
    main()

    