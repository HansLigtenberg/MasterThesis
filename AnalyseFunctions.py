"""
AnalyseFunctions.py

Purpose: Provides the functions to analyse the estimates

Version:
    1   Copied from Test8.py
        Change importanceDF to take list of combination matrices
    2   Change check for 'median' in the variable importance routines for check
        Number of dimensions of combinationMatrix
        Change standardise and importanceDf to also handle df with lists as index
    3   Create function for ABS plots in one figure
        Change combination names in importanceDf
    4   Change layout ABSN heatmap
    5   Include business recession indicator via ReadData3
    6   Make ABS curve accept matrices of ABShat as well
        Include filter in ABS curve and ABS one plot
    7   Give option to split contribution plots into three figures with in the 
    columns different models
    8   Add OLS function and JB test
    9   Change accuracy to something that makes more sense
    10  Function for plotting multiple heatmaps next to eachother
        Some testing for Shapley value importance
    11  Fix Shapley value importance
    12  Add function to only plot alpha and beta, not sigma
    13  Adapt histogram to let alpha not affect the edges

Author: Hans Ligtenberg
"""
import numpy as np
import Modules as mod
import matplotlib.pyplot as plt
import matplotlib as mpl
import ReadData3
import multiprocessing as mp
import pathos
import pickle
import dill
import os
import pandas as pd
import datetime
from scipy import stats
import scipy as sc
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

def prepareData(growth=True, standardiseMaxMin=False, a=-1, b=1, filterVars=None, filterValues=None):
    """Loads the data, select the correct columns and normalises it to have 
    mean zero and std one. When growth is True take the log-growth of INF and IP.
    By default standardise to have zero mean and unit standardeviation. Other
    option is to standardise such that all values lie between a and b."""
    df=ReadData3.loadDataRec()
    if filterVars is not None:
        df=df.drop(df[(df[filterVars]>filterValues).any(axis=1)].index)
    if growth:
        df['INF']=np.log(df['INF'])-np.log(df['INF'].shift(1))
        df['IP']=np.log(df['IP'])-np.log(df['IP'].shift(1))
    df[['HML','M']]=df[['HML','M']].shift(-1)
    df=df.drop(['VP 10', 'VP 20', 'VP 30', 'RF'], axis=1)
    df=df.dropna()
    df1=df[['HML', 'M']]
    df2=df.drop(['HML', 'M','REC'], axis=1)
    df3=df['REC']
    if standardiseMaxMin:
        df2=a+(df2-df2.min())*(b-a)/(df2.max()-df2.min())
    else:
        for col in df2.columns:
            df2[col]=(df2[col]-df2[col].mean())/df2[col].std()
    merged=pd.merge(df1,df2, how='outer',left_index=True, right_index=True)
    return pd.merge(merged,df3, how='outer',left_index=True, right_index=True)

def openAll(location, modelNrs):
    """Returns an array of loss k. Location is the full location except for the 
    number indicating the model number."""
    res=[]
    for i in modelNrs:
        with open(location+str(i), 'rb') as file:
                one=dill.load(file)
                res.append((i, float(one)))
    return dict(res)
    
def losses(index, location):
    """Returns the losses from the model at index. Location is the full 
    location except for the number indicating the model number. """
    with open(location+str(index), 'rb') as file:
        losses=dill.load(file)
    return losses

def parameters(index, location):
    """Returns parameters at the optimal k from the model at index. Location is 
    the full location except for the number indicating the model number. """
    with open(location+str(index), 'rb') as file:
        parameters=dill.load(file)
    return parameters

def hyperParameters(location, fileNameHP='Hyper-parameters', fileNameMSGS='Messages'):
    """Returns the hyper-parameters and the messages of the models. Location is
    the full location except for hyper-parameter and messages names."""
    with open(location+fileNameHP, 'rb') as f1, open(location+fileNameMSGS, 'rb') as f2:
        hyperParams=dill.load(f1)
        messages=dill.load(f2)
    return hyperParams, messages

def results(locHyperParameters='Results/Estimates/Hyper-Parameters', locMessages='Results/Estimates/Messages',
            locFinalLoss='Results/Estimates/Final_loss/Final_loss_', locK='Results/Estimates/k/k_',
            locLossesK='Results/Estimates/Loss_k/Loss_k_'):
    """Gets a data frame with results and hyper-parameters."""
    names=['Model', 'Architecture', 'Method', 'Loss', 'k', 'Loss k', 'Learning rate', 'Momentum', 'Batch size', 'Patience', 'Step size', 'Seed', 'Max updates', 'Rho']
    namesData=['Model', 'Train', 'Validation']
    table=[]
    data=[]
    missing=[]
    
    with open(locHyperParameters, 'rb') as f1, open(locMessages, 'rb') as f2:
        hyperParams=dill.load(f1)
        messages=dill.load(f2)
        
    for params in hyperParams:
        arch,method=params[:2]
        run=params[-1]
        B=params[4] #Only select models with B=128
        if B==128:
            row=[run, arch, method]
            if os.path.isfile(locK+str(run)):
                with open(locFinalLoss+str(run), 'rb') as finalLoss, open(locK+str(run), 'rb') as k, open(locLossesK+str(run), 'rb') as lossesK:
                    row.append(dill.load(finalLoss))
                    row.append(dill.load(k))
                    row.append(dill.load(lossesK))
                for val in params[2:-3]:
                    if callable(val):
                        row.append(messages[run][val.__name__])
                    else:
                        row.append(val)
                table.append(row)
                data.append([run]+params[-3:-1])
            else:
                missing.append(run)
    df=pd.DataFrame(table, columns=names)
    df.set_index('Model', inplace=True)
    dfData=pd.DataFrame(data, columns=namesData)
    dfData.set_index('Model', inplace=True)
    df[['Loss', 'k', 'Loss k']]=df[['Loss', 'k', 'Loss k']].astype(float)
    df=df.join(df['Architecture'].apply(pd.Series))
    df=df.rename(columns={0:'In', 1:'Layer 1', 2:'Layer 2', 3:'Layer 3', 4:'Layer 4', 5:'Layer 5'})
    if missing:
        print(f'Files for {missing} not found.')
    return df, dfData 

def lossCurve(model, locLosses, k, lossesK, style=None, ax=None, ylabel=None, xlabel=None, filt=np.inf):
    """Plots a loss curve."""
    if style is not None:
        plt.style.use(style)
    if ax is None:
        fig, ax=plt.subplots()
    lossesMin=losses(model, locLosses)
    ax.scatter(k[model]-1, lossesK[model], color='blue', s=7)
    ax.annotate(np.round(lossesK[model],4), (k[model]-1, lossesK[model]), horizontalalignment='right', verticalalignment='bottom')
    ax.set(xlabel=xlabel, ylabel=ylabel, title=str(model))
    lossesMin=[i if i<filt else None for i in lossesMin]
    ax.plot(lossesMin)
    
def addRecessions(ax, dates):
    """Shades the NBER recessions.
    Source https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions"""
    beginDate=dates[0]
    endDate=dates[-1]
    starts=['1948-11-1', '1953-7-1', '1957-8-1', '1960-4-1','1969-12-1', '1973-11-1',
            '1980-1-1', '1981-7-1', '1990-7-1', '2001-3-1', '2007-12-1', '2020-2-1']
    starts=[datetime.datetime.strptime(i, '%Y-%m-%d') for i in starts] 
    ends=['1949-10-1', '1954-5-1', '1958-4-1', '1961-2-1','1970-11-1', '1975-3-1',
          '1980-7-1', '1982-11-1', '1991-3-1', '2001-11-1','2009-6-1', '2021-5-1']
    ends=[datetime.datetime.strptime(i, '%Y-%m-%d') for i in ends]
    recessions=[]
    for i in range(min(len(starts), len(ends))):
        if starts[i]> beginDate and ends[i]<endDate:
            recessions.append([starts[i], ends[i]])
        elif starts[i]< beginDate and ends[i]<endDate:
            recessions.append([beginDate, ends[i]])
        elif starts[i]>beginDate and ends[i]>endDate:
            recessions.append([starts[i], endDate])
    for begin,end in recessions:
        ax.axvspan(begin, end, color='grey', alpha=0.5)

def ABSCurves(model, df, locP, dfData, style=None, axList=None, title=None,
              net=mod.FeedForwardLossLogSigma, color=None, alpha=None, absMatrix=None,
              filt=[[-np.inf,np.inf], [-np.inf,np.inf], [-np.inf,np.inf]]):
    """Gives the curves of the estimated alpha beta and sigma in the 3 ax
    provided in asList if not None. By default it is assumed log sigma is modelled."""
    if axList is None:
        axList=[]
        for i in range(3):
            fig,ax=plt.subplots()
            axList.append(ax)
    dates=dfData.index
    ylabels=['$\\hat\\alpha$', '$\\hat\\beta$', '$\\hat\\sigma$']
    if absMatrix is None:
        absHat=ABShat(model, df, locP, dfData, net=net)
    else:
        absHat=absMatrix
    if style is not None:
        plt.style.use(style)
    for i,(ax,ylabel,(filtLow,filtHigh)) in enumerate(zip(axList,ylabels,filt)):
        ax.set(ylabel=ylabel, title=str(model) if title is None else title)
        ax.axhline(0, linestyle='--', color='black')
        addRecessions(ax, dates)
        series=[k if filtLow<k<filtHigh else None for k in absHat[:,i]]
        ax.plot(dates, series, color=color, alpha=alpha)
    return absHat

def ABCurves(model, df, locP, dfData, style=None, axList=None, title=None,
              net=mod.FeedForwardLossLogSigma, color=None, alpha=None, absMatrix=None,
              filt=[[-np.inf,np.inf], [-np.inf,np.inf], [-np.inf,np.inf]]):
    """Gives the curves of the estimated alpha beta in the 2 ax
    provided in asList if not None. By default it is assumed log sigma is modelled."""
    if axList is None:
        axList=[]
        for i in range(2):
            fig,ax=plt.subplots()
            axList.append(ax)
    dates=dfData.index
    ylabels=['$\\hat\\alpha$', '$\\hat\\beta$']
    if absMatrix is None:
        absHat=ABShat(model, df, locP, dfData, net=net)
    else:
        absHat=absMatrix
    if style is not None:
        plt.style.use(style)
    for i,(ax,ylabel,(filtLow,filtHigh)) in enumerate(zip(axList,ylabels,filt)):
        ax.set(ylabel=ylabel, title=str(model) if title is None else title)
        ax.axhline(0, linestyle='--', color='black')
        addRecessions(ax, dates)
        series=[k if filtLow<k<filtHigh else None for k in absHat[:,i]]
        ax.plot(dates, series, color=color, alpha=alpha)
    return absHat

def plotStandardised(df, nonStandardised, style=None, axList=None, xlabel=None):
    """Plots the variables in df, except for the variables given by nonStandardised.
    Assumes the data in df is already standardised."""
    if axList is None:
        axList=[]
        for i in range(len(df.columns)-len(nonStandardised)):
            fig,ax=plt.subplots()
            axList.append(ax)
    if style is not None:
        plt.style.use(style)
    dates=df.index
    for label,ax in zip(df.drop(nonStandardised, axis=1).columns,axList):
        addRecessions(ax, dates)
        ax.plot(dates, df[label], label=label)
        ax.set(xlabel=xlabel, title=label+' standardised')
    
def ABShat(model, df, locParameters, dfData, factor='M', R='HML', 
           X=['DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP'], net=mod.FeedForwardLossLogSigma):
    """Returns the estimated alpha, beta and sigma. Note: for the parameters you can
    either fill in the final parameters or the parameters obtained while training.
    By default results are given for log sigma."""
    arch=df.at[model,'Architecture']
    network=net(arch)
    params=parameters(model, locParameters)
    network.setParameters(params)
    x=dfData[X].to_numpy()
    f=dfData[factor].to_numpy().reshape((-1,1))
    R=dfData[R].to_numpy().reshape((-1,1))
    inp=[[0,x]]
    addInp=[[len(arch),[['factor', f],['R', R]]]]
    ABS=network.sequentialOut(inp, len(arch)-1, 'output', addInp)
    if net==mod.FeedForwardLossLogSigma:
        ABS[:,2]=np.exp(ABS[:,2])
    elif net==mod.FeedForwardLoss:
        ABS[:,2]=abs(ABS[:,2])
    elif net==mod.FeedForwardT:
        ABS[:,2:]=np.exp(ABS[:,2:])
    else:
        raise Exception('Network not recognised.')
    return ABS

def contributionPlot(model, df, locParameters, dfData, axList=None, variable=None,
                     steps=250, factor='M', R='HML', recession=None, quantiles=[0.2, 0.4, 0.6, 0.8],
                     bins=25, net=mod.FeedForwardLossLogSigma, combinationMatrix=None, modelName=None):
    """Constructs a contribution plot for the varaible given or for all variables
    if it equals None, excluding the factor, return and if given recession indicator. By
    default assume log sigma is moddeled."""
    minData=dfData.min()
    maxData=dfData.max()
    if combinationMatrix is None:
        params=parameters(model, locParameters)
        arch=df.at[model,'Architecture']
        network=net(arch)
        network.setParameters(params)
    if recession is not None:
        dfDrop=dfData.drop([factor, R, recession], axis=1)
    else:
        dfDrop=dfData.drop([factor, R], axis=1)
    names=dfDrop.columns if variable is None else [variable]
    if axList is None:
        axList=[]
        for i in range(3*len(names)):
            fig,ax=plt.subplots()
            axList.append(ax)
    for i, name in enumerate(names):
        #Histogram
        axA=axList[i*3]
        axB=axList[i*3+1]
        axS=axList[i*3+2]
        density=stats.gaussian_kde(dfData[name])
        x=np.linspace(dfData[name].min(), dfData[name].max(), steps)
        for ax in axList[i*3:i*3+3]:
            ax2=ax.twinx()
            ax2.hist(dfData[name], color='tab:gray', alpha=0.3, bins=bins, density=True)
            ax2.plot(x, density(x), color='tab:gray', alpha=0.3, marker=' ')
            ax2.tick_params(right=False, labelright=False)
        #Contribution plots
        for q in quantiles:
            if combinationMatrix is None: #No combination
                X=dfDrop.quantile(q).to_numpy()
                X=np.resize(X, (steps, len(X)))
                X[:, dfDrop.columns.get_loc(name)]=np.linspace(minData[name], maxData[name], steps)
                inp=[[0,X]]
                addInp=[[len(arch),[['factor', np.ones((steps,1))],['R', np.ones((steps,1))]]]] #To make forward work
                ABS=network.sequentialOut(inp, len(arch)-1, 'output', addInp)
                if net==mod.FeedForwardLossLogSigma:
                    ABS[:,2]=np.exp(ABS[:,2])
                elif net==mod.FeedForwardLoss:
                    ABS[:,2]=abs(ABS[:,2])
                elif net==mod.FeedForwardT:
                    ABS[:,2:]=np.exp(ABS[:,2:])
                else:
                    raise Exception('Network not recognised.') 
            elif combinationMatrix.ndim==3: #Note: no matter what tensor is passed, median is used
                dfDataNew=copy.deepcopy(dfData)
                quants=dfDataNew.loc[:,(dfDataNew.columns!=R)&(dfDataNew.columns!=factor)].quantile(q).to_numpy()
                dfDataNew.loc[:,(dfDataNew.columns!=R)&(dfDataNew.columns!=factor)]=quants
                dfDataNew=dfDataNew[:steps]
                dfDataNew[name]=np.linspace(minData[name], maxData[name], steps)
                ABSTensor=np.zeros((len(model),steps,3))
                for i,m in enumerate(model):
                    ABSTensor[i,:,:]=ABShat(m, df, locParameters, dfDataNew, factor=factor, R=R, net=net)
                ABS=np.median(ABSTensor,0)
            else: #Combination
                if combinationMatrix.ndim>2: raise Exception('Currently only combination matrices and vectors are supported.')
                dfDataNew=copy.deepcopy(dfData)
                quants=dfDataNew.loc[:,(dfDataNew.columns!=R)&(dfDataNew.columns!=factor)].quantile(q).to_numpy()
                dfDataNew.loc[:,(dfDataNew.columns!=R)&(dfDataNew.columns!=factor)]=quants
                dfDataNew=dfDataNew[:steps]
                dfDataNew[name]=np.linspace(minData[name], maxData[name], steps)
                if combinationMatrix.ndim==1 or 1 in combinationMatrix.shape:
                    combinationMatrix=np.resize(combinationMatrix, (steps,len(model))).T
                ABSTensor=np.zeros((len(model),steps,3))
                for i,m in enumerate(model):
                    ABSTensor[i,:,:]=ABShat(m, df, locParameters, dfDataNew, factor=factor, R=R, net=net)
                ABS=np.zeros((steps,3))
                for t in range(steps):
                    for j in range(len(model)):
                        ABS[t,:]+=combinationMatrix[j,t]*ABSTensor[j,t,:]

            axA.plot(x,ABS[:,0], label=str(q))
            axB.plot(x,ABS[:,1], label=str(q))
            axS.plot(x,ABS[:,2], label=str(q))
            if modelName is None:
                axA.set(title=f'{name} for'+' $\\hat\\alpha$'+f' in {model}')
                axB.set(title=f'{name} for'+' $\\hat\\beta$'+f' in {model}')
                axS.set(title=f'{name} for'+' $\\hat\\sigma$'+f' in {model}')
            else:
                axA.set(title=f'{name} for'+' $\\hat\\alpha$'+f' in {modelName}')
                axB.set(title=f'{name} for'+' $\\hat\\beta$'+f' in {modelName}')
                axS.set(title=f'{name} for'+' $\\hat\\sigma$'+f' in {modelName}')
    return ABS
      
def lossNormal(alpha, beta=None, sigma=None, R=None, factor=None, fracTrain=0.8, sample='validation'):
    """Gives the average loss of the loss function according to negative log normal. 
    Alpha is either a T vector or a 3xT matrix of alpha or alpah,beta,sigma. Not
    log sigma. If fracTrain is not 1 all variables are split in a training and
    validation set. The desired sample is then selected via sample."""
    if alpha.shape[1]>=3:
        R=beta if R is None else R
        factor=sigma if factor is None else factor
        beta=alpha[:,1]
        sigma=alpha[:,2]
        alpha=alpha[:,0]
    elif R is None or factor is None:
        raise Exception('R and factor must be given.')
    if fracTrain!=1:
        train=int(len(R)*fracTrain)
        if sample=='train':
            alpha=alpha[:train]
            beta=beta[:train]
            sigma=sigma[:train]
            R=R[:train]
            factor=factor[:train]
        elif sample=='validation':
            alpha=alpha[train:]
            beta=beta[train:]
            sigma=sigma[train:]
            R=R[train:]
            factor=factor[train:]  
        else: raise Exception('Sample selection not recognised.')
    cumulative=0
    for a,b,s,R,f in zip(alpha, beta, sigma, R, factor):
        cumulative+=0.5*np.log(2*np.pi)+np.log(abs(s))+(R-a-f*b)**2/(2*s**2)
    return cumulative/len(alpha)

def combineInputs(input1, input2):
    """Combines two input data sets into one. Data sets must have the same
    structure."""
    for index,pair in enumerate(input2):
        matrix=np.vstack((input1[index][1], pair[1]))
        input1[index][1]=matrix
    return input1

def combineAdditionalInputs(addInput1, addInput2):
    """Combines two additional input data sets into one. Data sets must have the
    same structure."""
    for index,l in enumerate(addInput2):
        for i,pair in enumerate(l[1]):
            matrix=np.vstack((addInput1[index][1][i][1],pair[1]))
            addInput1[index][1][i][1]=matrix
    return addInput1        
            
def selectSample(model, dfTrainVal, sample='whole'):
    """Selects the correct samples."""
    inpVal, addInpVal=dfTrainVal.loc[model,['Train', 'Validation']]
    if sample=='whole':
        inpTrain, addInpTrain=copy.deepcopy(dfTrainVal.loc[model,'Train'])
        inpVal, addInpVal=copy.deepcopy(dfTrainVal.loc[model,'Validation'])
        inputs=combineInputs(inpTrain,inpVal)
        addInputs=combineAdditionalInputs(addInpTrain,addInpVal)
    elif sample=='train':
        inputs, addInputs=copy.deepcopy(dfTrainVal.loc[model,'Train'])
    elif sample=='validation':
        inputs, addInputs=copy.deepcopy(dfTrainVal.loc[model,'Validation'])
    return inputs, addInputs
            
def variableImportance(model, df, locParameters, dfTrainVal, relative=True, sample='whole',
                       net=mod.FeedForwardLossLogSigma, combinationMatrix=None, dfData=None,
                       R='HML', factor='M', lossFunction=lossNormal):
    """Gives the rise in loss when one variable is set to zero and the rest 
    remains unchanged. By default relative importance is used. By default
    assume log sigma is moddeled."""
    
    if combinationMatrix is None: #No combination
        inpVal, addInpVal=selectSample(model, dfTrainVal, sample)
        params=parameters(model, locParameters)
        arch=df.at[model,'Architecture']
        network=net(arch)
        network.setParameters(params)
        numberOfVariables=inpVal[0][1].shape[1]
        lossOriginal=network.loss(network.longInput(inpVal), len(arch), 'output', network.longAddInput(addInpVal))
        res=[]
        for i in range(numberOfVariables):
            inpZeroed=copy.deepcopy(inpVal)
            inpZeroed[0][1][:,i]=0
            res.append(network.loss(network.longInput(inpZeroed), len(arch), 'output', network.longAddInput(addInpVal)))
    elif combinationMatrix.ndim==3: #Median
        res=[]
        inputs, addInputs=selectSample(model[0], dfTrainVal, sample)
        numberOfVariables=inputs[0][1].shape[1]
        T=len(inputs[0][1])
        N=len(model)
        for i in range(numberOfVariables):
            dfDataZeroed=copy.deepcopy(dfData)
            dfDataZeroed.iloc[:,2+i]=0
            ABSTensor=np.zeros((N,T,3))
            for i,m in enumerate(model):
                ABSTensor[i,:,:]=ABShat(m, df, locParameters, dfDataZeroed, factor=factor, R=R, net=net)
            ABSCombined=np.median(ABSTensor,0)
            res.append(lossFunction(ABSCombined, R=dfDataZeroed[R], factor=dfDataZeroed[factor], fracTrain=1))
        for i,m in enumerate(model):
            ABSTensor[i,:,:]=ABShat(m, df, locParameters, dfData, factor=factor, R=R, net=net)
        ABSCombined=np.median(ABSTensor,0)
        lossOriginal=lossFunction(ABSCombined, R=dfData[R], factor=dfData[factor], fracTrain=1)
    else: #Combination
        if combinationMatrix.ndim>2: raise Exception('Currently only combination matrices and vectors are supported.')
        res=[]
        inputs, addInputs=selectSample(model[0], dfTrainVal, sample)
        numberOfVariables=inputs[0][1].shape[1]
        T=len(inputs[0][1])
        N=len(model)
        if combinationMatrix.ndim==1 or 1 in combinationMatrix.shape:
            combinationMatrix=np.resize(combinationMatrix, (len(inputs[0][1]),len(model))).T
        for i in range(numberOfVariables):
            dfDataZeroed=copy.deepcopy(dfData)
            dfDataZeroed.iloc[:,2+i]=0
            ABSTensor=np.zeros((N,T,3))
            for i,m in enumerate(model):
                ABSTensor[i,:,:]=ABShat(m, df, locParameters, dfDataZeroed, factor=factor, R=R, net=net)
            ABSCombined=np.zeros((T,3))
            for t in range(T):
                for j in range(N):
                    ABSCombined[t,:]+=combinationMatrix[j,t]*ABSTensor[j,t,:]
            res.append(lossFunction(ABSCombined, R=dfDataZeroed[R], factor=dfDataZeroed[factor], fracTrain=1))
        ABSTensor=np.zeros((N,T,3))
        for i,m in enumerate(model):
            ABSTensor[i,:,:]=ABShat(m, df, locParameters, dfData, factor=factor, R=R, net=net)
        ABSCombined=np.zeros((T,3))
        for t in range(T):
            for j in range(N):
                ABSCombined[t,:]+=combinationMatrix[j,t]*ABSTensor[j,t,:]
        lossOriginal=lossFunction(ABSCombined, R=dfData[R], factor=dfData[factor], fracTrain=1)    
       
    if relative:
        total=sum(res-lossOriginal)
        return (res-lossOriginal)/total
    else:
        return res

def SSD(model, df, locParameters, dfTrainVal, relative=True, output='loss', 
        sample='whole', inputName='input', X=['DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP'],
        net=mod.FeedForwardLossLogSigma, combinationMatrix=None):
    """Gives the sum of squared partial derivatives of the output with respect
    to each input variable. Scaling by input variance is ommitted, since inputs
    are assumed to be standardised. Sample indicates which part of the data 
    must be used. By default relative importance is used. By default assume log
    sigma is moddeled."""
    model=model if isinstance(model,list) else [model]
    inputs,addInputs=selectSample(model[0], dfTrainVal, sample)
    T=len(inputs[0][1])
    if combinationMatrix is None:
        combinationMatrix=np.ones((1,T))
    elif combinationMatrix.ndim==1 or 1 in combinationMatrix.shape:
        combinationMatrix=np.resize(combinationMatrix, (T,len(model))).T
    
    if combinationMatrix.ndim<=2:
        cumulativeT=0
        for t in range(T):
            cumulativeJ=0
            for j,m in enumerate(model):
                params=parameters(m, locParameters)
                arch=df.at[m,'Architecture']
                network=net(arch)
                network.setParameters(params)
                inputs,addInputs=selectSample(m, dfTrainVal, sample)
                inputs=network.longInput(inputs)
                addInputs=network.longAddInput(addInputs)
                out=len(arch) if output=='loss' else len(arch)-1
                network.forward(inputs[t], addInputs[t])
                cumulativeJ+=combinationMatrix[j,t]*network.backward(out, 0, inputName)
            cumulativeT+=cumulativeJ**2
    elif output=='loss':
        if net!=mod.FeedForwardLossLogSigma:
            raise Exception('Network not implemented.')
        cumulativeT=0
        for t in range(T):
            cumulativeJ=0
            for j,m in enumerate(model):
                omegaSigmaSigma=0
                for k,mm in enumerate(model):
                    params=parameters(mm, locParameters)
                    arch=df.at[mm,'Architecture']
                    network=net(arch)
                    network.setParameters(params)
                    inputs,addInputs=selectSample(mm, dfTrainVal, sample)
                    inputs=network.longInput(inputs)
                    addInputs=network.longAddInput(addInputs)
                    out=len(arch)
                    network.forward(inputs[t], addInputs[t])  
                    omegaSigmaSigma+=combinationMatrix[2,k,t]*network.nodes[out-1].outputs['output'][2]
                params=parameters(m, locParameters)
                arch=df.at[m,'Architecture']
                network=net(arch)
                network.setParameters(params)
                inputs,addInputs=selectSample(m, dfTrainVal, sample)
                inputs=network.longInput(inputs)
                addInputs=network.longAddInput(addInputs)
                out=len(arch)
                network.forward(inputs[t], addInputs[t])
                dABSdX=network.backward(out-1,0,inputName)
                dJdABS=network.backward(out,out,'input')
                for l in range(len(dABSdX)):
                    if l!=2:
                        cumulativeJ+=dABSdX[l]*dJdABS[0][l]*combinationMatrix[l,j,t]
                    else:
                        cumulativeJ+=dABSdX[l]*dJdABS[0][l]*combinationMatrix[l,j,t]*(1/omegaSigmaSigma)
            cumulativeT+=cumulativeJ**2
    else:
        if net!=mod.FeedForwardLossLogSigma:
            raise Exception('Network not implemented.')
        cumulativeT=0
        for t in range(T):
            cumulativeJ=0
            for j,m in enumerate(model):
                params=parameters(m, locParameters)
                arch=df.at[m,'Architecture']
                network=net(arch)
                network.setParameters(params)
                inputs,addInputs=selectSample(m, dfTrainVal, sample)
                inputs=network.longInput(inputs)
                addInputs=network.longAddInput(addInputs)
                out=len(arch)-1
                network.forward(inputs[t], addInputs[t])
                dABSdX=network.backward(out,0,inputName)
                sigmaHat=network.nodes[out].outputs['output'][2]
                for l in range(len(dABSdX)):
                    if l!=2:
                        dABSdX[l]=dABSdX[l]*combinationMatrix[l,j,t]
                    else:
                        dABSdX[l]=dABSdX[l]*combinationMatrix[l,j,t]*sigmaHat
                cumulativeJ+=dABSdX
            cumulativeT+=cumulativeJ**2            
        
    res=((1/len(inputs))*cumulativeT).squeeze()
    
    if relative:
        total=sum(res)
        return res/total
    else:
        return res

def powerset(l):
    """Gives a generator for the powerset of list l. Empty part is ommitted."""
    if len(l)<=0:
        yield []
    if len(l)==1:
        yield l
        yield []
    else:
        for item in powerset(l[1:]):
            yield [l[0]]+item
            yield item

def v(S, network, outputNodeNr, inputs, addInputs=None, draws=10, rng=None, 
      outputName='output', net=mod.FeedForwardLossLogSigma, combinationMatrix=None, 
      lossFunction=lossNormal, df=None, locParameters=None, dfData=None,
      factor='M', R='HML', model=None):
    """Approximate the loss when there is an intervention on the variables in 
    S. The intervention consists of drawing the variables from a multivariate
    normal distribution with mean equal to the sample mean and covariance equal
    to the sample covaraince matrix. The approximation is based on draws draws.
    Note: not completely general."""
    if rng is None:
        rng=np.random.default_rng(19980528)
    if combinationMatrix is None:
        lAddInputs=network.longAddInput(addInputs)
    matrix=inputs[0][1]
    intervene=matrix[:,list(S)]
    cov=np.cov(intervene.T)
    mean=np.mean(intervene,0)
    loss=0
    for i in range(draws):
        newInputs=inputs
        if np.size(cov)<=1:
            newColumns=mean+np.sqrt(cov)*rng.normal(0,1,(len(matrix),1))
        else:
            newColumns=mean+(np.linalg.cholesky(cov)@rng.normal(0,1,(len(cov),len(matrix)))).T
        for j,k in enumerate(S):
            newInputs[0][1][:,k]=newColumns[:,j]
        if combinationMatrix is None: #No combination
            loss+=network.loss(network.longInput(newInputs), outputNodeNr, outputName, lAddInputs)
        elif combinationMatrix.ndim==3: #Median combination
            dfDataNew=copy.deepcopy(dfData)
            dfDataNew.iloc[:,2:]=newInputs[0][1]
            T=len(newInputs[0][1])
            N=len(model)
            ABSTensor=np.zeros((N,T,3))
            for i,oneModel in enumerate(model):
                ABSTensor[i,:,:]=ABShat(oneModel, df, locParameters, dfDataNew, factor=factor, R=R, net=net)
            ABSCombined=np.median(ABSTensor,0)
            loss+=lossFunction(ABSCombined, R=addInputs[0][1][1][1], factor=addInputs[0][1][0][1], fracTrain=1)
        else: #Combination
            if combinationMatrix.ndim>2: raise Exception('Currently only combination matrices and vectors are supported.')
            dfDataNew=copy.deepcopy(dfData)
            dfDataNew.iloc[:,2:]=newInputs[0][1]
            T=len(newInputs[0][1])
            N=len(model)
            ABSTensor=np.zeros((N,T,3))
            for i,oneModel in enumerate(model):
                ABSTensor[i,:,:]=ABShat(oneModel, df, locParameters, dfDataNew, factor=factor, R=R, net=net)
            if combinationMatrix.ndim==1 or 1 in combinationMatrix.shape:
                combinationMatrix=np.resize(combinationMatrix, (T,N)).T
            ABSCombined=np.zeros((T,3))
            for t in range(T):
                for j in range(N):
                    ABSCombined[t,:]+=combinationMatrix[j,t]*ABSTensor[j,t,:]
            loss+=lossFunction(ABSCombined, R=addInputs[0][1][1][1], factor=addInputs[0][1][0][1], fracTrain=1)
                
    return loss/draws   
        
def ShapleyImportance(model, df, locParameters, dfTrainVal, locShapley=os.getcwd()+'/Results/Estimates/Shapley', 
                      relative=True, draws=10, rng=None, outputName='output', sample='whole',
                      net=mod.FeedForwardLossLogSigma, combinationMatrix=None, lossFunction=lossNormal,
                      factor='M', R='HML', dfData=None):
    """Gives the Shapley value of variable importance. By default the relative
    importance is given. The non-relative value is stored. Note: function is 
    not written for arbitrary data sets. by default assume log sigma is moddeled."""
    try:
        with open(locShapley+f'/Shapley_{model}_draws_{draws}_{outputName}_{sample}', 'rb') as file:
            res=dill.load(file)
    except:
        if combinationMatrix is None:
            params=parameters(model, locParameters)
            arch=df.at[model,'Architecture']
            network=net(arch)
            network.setParameters(params)
            inputs,addInputs=selectSample(model, dfTrainVal, sample)
        else:
            if len(model)!=len(combinationMatrix): raise Exception('Number of models and weights do not match.')
            inputs,addInputs=selectSample(model[0], dfTrainVal, sample)
            network=None
            arch=[]
        
        if rng is None:
            rng=np.random.default_rng(19980528)
            
        n=inputs[0][1].shape[1]
        res=np.zeros(n)
        l=[i for i in powerset(list(range(n)))]
        l.sort()
        for S in l[1:]:
            print(S)
            vS=v(S, network, len(arch), inputs, addInputs=addInputs, draws=draws, rng=rng,
                 outputName=outputName, net=net, combinationMatrix=combinationMatrix,
                 lossFunction=lossFunction, df=df, locParameters=locParameters, dfData=dfData,
                 factor=factor, R=R, model=model)
            for i in set(range(n))-set(S):
                Si=S+[i]
                Si.sort()
                vSi=v(Si, network, len(arch), inputs, addInputs=addInputs, draws=draws, rng=rng,
                      outputName=outputName, net=net, combinationMatrix=combinationMatrix,
                      lossFunction=lossFunction, df=df, locParameters=locParameters, dfData=dfData,
                      factor=factor, R=R, model=model)
                res[i]+=np.math.factorial(len(S))*np.math.factorial(n-len(S)-1)/np.math.factorial(n)*(vS-vSi)
        with open(locShapley+f'/Shapley_{model}_draws_{draws}_{outputName}_{sample}', 'wb') as file:
            dill.dump(res, file)
        
    if relative:
        total=sum(res)
        return res/total
    else:
        return res
    
def importanceDf(model, df, locParameters, dfTrainVal, method='importance', inputName='input', 
                 locShapley=os.getcwd()+'/Results/Estimates/Shapley', relative=True, draws=10, 
                 rng=None, outputSSD='loss', outputName='output', sample='whole', 
                 X=['DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP'], net=mod.FeedForwardLossLogSigma,
                 combinationMatrix=None, lossFunction=lossNormal, factor='M', R='HML', dfData=None, combinationNames=None):
    """Returns a data frame with the importance measures according to method for
    the models in model. Shapley value is calculated in parallel using Pathos.
    By default it is assumed log sigma is moddeled. If combinationMatrix is not
    None the combination constructed by those weights is used."""
    model=model if isinstance(model, list) else [model]
    combinationMatrix=combinationMatrix if isinstance(combinationMatrix,list) else [None for i in model]
    if sample!='whole' and isinstance(model[0], list):
        raise Exception('Other samples than whole are currently not supported for model combinations.')
    if method=='importance':
        table=[]
        for i,combMat in zip(model,combinationMatrix):
            table.append(variableImportance(i, df, locParameters, dfTrainVal, relative=relative, sample=sample,
                                            net=net, combinationMatrix=combMat, dfData=dfData,
                                            R=R, factor=factor, lossFunction=lossFunction).squeeze())
            modelNames=[]
            j=0
            for i in model:
                if isinstance(i, int):
                    modelNames.append(i)
                elif combinationNames is None:
                    modelNames.append(str(i))
                else:
                    modelNames.append(combinationNames[j])
                    j+=1
            index=pd.Index(modelNames, name='Model')
    elif method=='SSD':
        if outputSSD=='loss':
            table=[]
            for i,combMat in zip(model,combinationMatrix):
                table.append(SSD(i, df, locParameters, dfTrainVal, relative=relative, output=outputSSD,
                                 sample=sample, inputName=inputName, net=net, combinationMatrix=combMat).squeeze())
            modelNames=[]
            j=0
            for i in model:
                if isinstance(i, int):
                    modelNames.append(i)
                elif combinationNames is None:
                    modelNames.append(str(i))
                else:
                    modelNames.append(combinationNames[j])
                    j+=1
            index=pd.Index(modelNames, name='Model')
        else:
            table=SSD(model[0], df, locParameters, dfTrainVal, relative=relative, output=outputSSD,
                      sample=sample, inputName=inputName, net=net, combinationMatrix=combinationMatrix[0])
            numberOfOutputs=len(table)
            outputNames=['$\\alpha$', '$\\beta$', '$s$', '$n$']
            for i,combMat in zip(model[1:],combinationMatrix[1:]):
                table=np.vstack(((table, SSD(i, df, locParameters, dfTrainVal, relative=relative,
                                             output=outputSSD, sample=sample, inputName=inputName, net=net,
                                             combinationMatrix=combMat))))
            modelNames=[]
            j=0
            for i in model:
                if isinstance(i, int):
                    modelNames.append(i)
                elif combinationNames is None:
                    modelNames.append(str(i))
                else:
                    modelNames.append(combinationNames[j])
                    j+=1
            index=pd.MultiIndex.from_product([modelNames, outputNames[:numberOfOutputs]], names=['Model', 'Parameter'])
    elif method=='Shapley':
        n=len(model)        
        pool=pathos.multiprocessing.ProcessingPool(maxtasksperchild=1)             
        table=pool.map(ShapleyImportance, model, [df]*n, [locParameters]*n, [dfTrainVal]*n,
                       [locShapley]*n, [relative]*n, [draws]*n, [rng]*n, [outputName]*n,
                       [sample]*n, [net]*n, combinationMatrix, [lossFunction]*n,
                       [factor]*n, [R]*n, [dfData]*n)
        pool.close()
        pool.join()
        pool.clear()
        modelNames=[]
        j=0
        for i in model:
            if isinstance(i, int):
                modelNames.append(i)
            elif combinationNames is None:
                modelNames.append(str(i))
            else:
                modelNames.append(combinationNames[j])
                j+=1
        index=pd.Index(modelNames, name='Model')
    else: raise Exception('Method not recognised.')
    return pd.DataFrame(table, index=index, columns=X)
    
def filterParameter(hyperParams, variableNr, l):
    """Return a list of models for which the variable under variableNr is in l."""
    l= l if isinstance(l, list) else [l]
    res=[]
    for i in hyperParams:
        if i[variableNr] in l:
            res.append(i[-1])
    return res

def filterDict(d, keys):
    """Returns a dictionary from d with keys in keys."""
    res={}
    for key, value in zip(d.keys(), d.values()):
        if key in keys:
            res[key]=value
    return res    

def scatter(dataX, dataY, ax=None, OLS=True, label=None, xlabel=None, ylabel=None, title=None):
    """Gives a scatter plot of the losses during first round of training against 
    the retrained losses."""
    if ax is None:
        fig,ax=plt.subplots()
    ax.scatter(dataX, dataY, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if OLS:
        x=np.vstack((np.ones(len(dataX)),dataX)).T
        b0,b1=np.linalg.lstsq(x, dataY, None)[0]
        ax.plot(dataX, b0+np.array(dataX)*b1, color='blue', marker='None', linewidth=1)
        return ax, b0, b1
    return ax

def histogram(data, ax=None, density=True, drawDensity=True, mean=True, bins=30, 
              label=None, xlabel=None, ylabel=None, title=None, alpha=0.5, densityExtend=0.1,
              labelMean=None, linewidth=0.5):
    if ax is None:
        fig,ax=plt.subplots()
    try:
        color=next(ax._get_patches_for_fill.prop_cycler)['color']
    except:
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    ax.hist(data, bins=bins, label=label, density=density, edgecolor=color,
            fc=mpl.colors.to_rgba(color,alpha), linewidth=linewidth)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if drawDensity:
        density=stats.gaussian_kde(data)
        x=np.linspace(data.min()-densityExtend, data.max()+densityExtend, 200)
        ax.plot(x, density(x), color=color, alpha=1, marker=' ')
    if mean:
        ax.axvline(data.mean(), color=color, linestyle='--', label=labelMean)
        return ax

def intfmt(number):
    if isinstance(number,str):
        return number
    return str(int(np.round(number)))

def nonfmt(string):
    return string

def decfmt(number, places=4):
    if np.isinf(number):
        return '$\infty$'
    elif number<=9999:
        return str(np.round(number, places))
    elif number>9999:
        s='%.*e'%(places, number)
        mantissa, exponent=s.split('e+')
        return f'${mantissa}\\cdot 10^{{{exponent}}}$'
    
def table(df, name, locTable, formatters=None, tex=True, locCaption=None, escape=False, 
          column_format=None, na_rep='', index=False, longtable=False, multirow=True, float_format=None, header=True):
    """Prints the data frame as a talbe. Either as a tex or txt file. If written as
    tex file, the caption is read from a file."""
    locCaption=locTable+'/Captions' if locCaption is None else locCaption
    if tex:
        with open(locTable+'/'+name+'.tex', 'w') as tex:
            try:
                with open(locCaption+'/'+name+'.txt', 'r') as cap:
                    caption=cap.read()
            except:
                print(f'No caption found for {name}.')
                caption=None
            df.to_latex(buf=tex, na_rep=na_rep, formatters=formatters, escape=escape,
                  longtable=longtable, index=index, column_format=column_format, caption=caption,
                  label='tab:'+name, multirow=multirow, float_format=float_format, header=header)
    else:
        with open(locTable+'/'+name+'.txt', 'w') as txt:
           df.to_string(buf=txt, na_rep=na_rep, formatters=formatters, index=index, header=header)
    return

def selectArch(df, arch, layerNames=['In', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 5']):
    """Returns the data frame with the selected archtectures."""
    arch=arch if isinstance(arch[0], list) else [arch]
    maskTotal=(df[layerNames[0]]==arch[0][0])
    for w,name in zip(arch[0][1:],layerNames[1:]):
        maskTotal=(maskTotal&(df[name]==w))
    for a in arch[1:]:
        mask=(df[layerNames[0]]==a[0])
        for w,name in zip(a[1:],layerNames[1:]):
            mask=(mask&(df[name]==w))
        maskTotal=(maskTotal|mask)
    return df[maskTotal]

def heatmap(df, ax=None, xlabel=None, ylabel=None, cbarlabel='Relative importance', 
            variableNames=None, models=None, title=None):
    if ax is None:
        fig, ax=plt.subplots()
    if variableNames is None:
        variableNames=df.columns
    if models is None:
        models=df.index
    obj=ax.imshow(df) 
    ax.set_xticks(np.arange(len(variableNames)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(variableNames)
    ax.set_yticklabels(models)
    ax.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
    cbar=ax.figure.colorbar(obj, ax=ax, shrink=1, aspect=50)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_xlabel(xlabel)    
    ax.xaxis.set_label_position('top') 
    ax.set_ylabel(ylabel)

    ax.set_title(title)
    
def heatmapMultiple(dfList, xlabels=None, ylabels=None, cbarlabel='Relative importance',
                    variableNames=None, models=None, titles=None):
    """Plots the heatmaps for the df in dfList. Sets only one colorbar."""
    n=len(dfList)
    fig,axes=plt.subplots(1, n, constrained_layout=True, tight_layout=False)
    xlabels=n*[None] if xlabels is None else xlabels
    ylabels=n*[None] if ylabels is None else ylabels
    titles=n*[None] if titles is None else titles
    
    for df,ax,xlabel,ylabel,title in zip(dfList,axes,xlabels,ylabels,titles):
        if variableNames is None:
            variableNames=df.columns
        if models is None:
            models=[i for i in df.index]
        ims=ax.imshow(df) 
        ax.set_xticks(np.arange(len(variableNames)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(variableNames)
        ax.set_yticklabels(models)
        ax.set_xlabel(xlabel)    
        ax.xaxis.set_label_position('top') 
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        if any(isinstance(val, str) for val in df.index.get_level_values(0)):
            plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cbar=fig.colorbar(ims, ax=axes, shrink=0.4)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    return fig, fig.axes
    
def ABSNHeatmap(df, xlabel=None, ylabel=None, cbarlabel='Relative importance', variableNames=None, models=None, title=None):
    params=df.index.levels[1]
    numberOfParameters=len(params)
    fig=plt.figure(constrained_layout=True, tight_layout=False)
    widths=[10 for i in range(numberOfParameters)]+[1]
    spec=fig.add_gridspec(ncols=numberOfParameters+1, nrows=1, width_ratios=widths)
    if variableNames is None:
        variableNames=df.columns
    if models is None:
        models=[df.index[i][0] for i in range(0, len(df.index), numberOfParameters)]
    for i in range(numberOfParameters):
        ax=fig.add_subplot(spec[0,i])
        ims=ax.imshow(df.xs(params[i], level=1, axis=0)) 
        ax.set_xticks(np.arange(len(variableNames)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(variableNames)
        ax.set_yticklabels(models)
        ax.set_xlabel(xlabel)    
        ax.xaxis.set_label_position('top') 
        ax.set_ylabel(ylabel)
        ax.set_title(params[i])
        ax.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        if any(isinstance(val, str) for val in df.index.get_level_values(0)):
            plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cbar=fig.colorbar(ims, cax=fig.add_subplot(spec[0,-1]))
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    return fig, fig.axes

def ABSNHeatmap2(df, xlabel=None, ylabel=None, cbarlabel='Relative importance', variableNames=None, models=None, title=None):
    # plt.rcParams['figure.constrained_layout.use']=True
    params=df.index.levels[1]
    numberOfParameters=len(params)
    fig,axes=plt.subplots(1,numberOfParameters, constrained_layout=True, tight_layout=False)
    if variableNames is None:
        variableNames=df.columns
    if models is None:
        models=[df.index[i][0] for i in range(0, len(df.index), numberOfParameters)]
    for i,ax in enumerate(axes):
        ims=ax.imshow(df.xs(params[i], level=1, axis=0)) 
        ax.set_xticks(np.arange(len(variableNames)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(variableNames)
        ax.set_yticklabels(models)
        ax.set_xlabel(xlabel)    
        ax.xaxis.set_label_position('top') 
        ax.set_ylabel(ylabel)
        ax.set_title(params[i])
        ax.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        if any(isinstance(val, str) for val in df.index.get_level_values(0)):
            plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    cbar=fig.colorbar(ims, ax=axes, shrink=0.4)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    return fig, fig.axes
    
def standardise(df, a=-1, b=1):
    """Standardises the dataframe such that each row lies between a and b."""
    dfNew=copy.deepcopy(df)
    for i,row in enumerate(df.itertuples()):
        minRow=min(row[1:])
        maxRow=max(row[1:])
        for j, value in enumerate(row[1:]):
            dfNew.iat[i,j]=a+(value-minRow)*(b-a)/(maxRow-minRow)
    return dfNew

def saveFigure(fig, name, locFigure=os.getcwd()+'/Results/Figures', backend='pgf', 
               width=418/72.27, height=None, fraction=1, subplots=(1,1), textheightIn=591.5302/72.27):
    """Saves the figure after width and height of the figure are set. By default
    height is set to golden ratio of width."""
    if height is None:
        fig.set_size_inches(width*fraction, min((5**0.5-1)/2*width*fraction*(subplots[0]/subplots[1]), 0.95*textheightIn))
    else:
        fig.set_size_inches(width*fraction, height*fraction)
    if backend=='pgf':
        fig.savefig(locFigure+'/'+name+'.pgf', backend=backend)
    else:
        fig.savefig(locFigure+'/'+name, backend=backend)
        
def modelsAdd(df, locParameters, dfData, addCombination='mean', lossType='Loss k',
              lossFunction=lossNormal, net=mod.FeedForwardLossLogSigma, p=5,
              R='HML', factor='M', fracTrain=0.8, sample='validation'):
    """Considers adding models for model averaging until the loss over sample 
    has not decreased for p successive models, where the models are sorted based
    on lossType and averaged using combination."""
    candidates=df.sort_values(lossType).index
    selected=[candidates[0]]
    absn=ABShat(selected[0], df, locParameters, dfData, net=net)
    minLoss=lossFunction(absn, R=dfData[R], factor=dfData[factor], fracTrain=fracTrain, sample=sample)
    j=0
    length=-1
    while length<len(selected):    
        for model in candidates[1:]:
            absn,previousModels=modelAveraging(selected+[model], df, locParameters, dfData, addCombination, net=net)
            loss=lossFunction(absn, R=dfData[R], factor=dfData[factor], fracTrain=fracTrain, sample=sample)
            if loss<minLoss:
                minLoss=loss
                selected.append(model)
                j=0
            else:
                j+=1
                if j>=p:
                    break
        length=len(selected)
    return selected
    
def combinationAccuracy(absn, dfData, lossFunction=lossNormal, R='HML', factor='M', fracTrain=0.8, sample='validation'):
    """Combines the matrices in absn weighted using the accuracy of each matrix"""
    losses=np.zeros(len(absn))
    for i,matrix in enumerate(absn):
        losses[i]=lossFunction(matrix, R=dfData[R], factor=dfData[factor], fracTrain=fracTrain, sample=sample)
    accuracy=(min(losses)/losses)/sum(min(losses)/losses)
    return np.tensordot(accuracy, absn, axes=1), accuracy

def combinationInverseLoss(absn, dfData, lossFunction=lossNormal, R='HML', factor='M', fracTrain=0.8, sample='validation', k=1):
    """Combines the matrices in absn weighted using the inverse loss raised to the power k of each matrix"""
    losses=np.zeros(len(absn))
    for i,matrix in enumerate(absn):
        losses[i]=lossFunction(matrix, R=dfData[R], factor=dfData[factor], fracTrain=fracTrain, sample=sample)
    weights=(1/losses**k)/sum(1/losses**k)
    return np.tensordot(weights, absn, axes=1), weights
        
def modelAveraging(models, df, locParameters, dfData, combination, net=mod.FeedForwardLossLogSigma,
                   dictModel={}, dictCombination={}):
    """Combines the models as selected by models and combines their output via
    combination."""
    extraOut={}
    #Select models
    if isinstance(models, list):
        extraOut['models']=models
    elif models=='add':
        models=modelsAdd(df, locParameters, dfData, **dictModel)
        extraOut['models']=models
    else: raise Exception('Models selection not recognised.')
    
    #Calculate output
    n=len(models)
    absn=ABShat(models[0], df, locParameters, dfData, net=net)
    absn=np.tile(absn,(n,1,1)) #first dimension is model nummer
    for i,model in enumerate(models[1:]):
        absn[i+1,:,:]=ABShat(model, df, locParameters, dfData, net=net)
    
    #Combine outputs
    if combination=='mean':
        combined=np.mean(absn,0)
        extraOut['combinationMatrix']=np.full(int(n),1/n)
    elif combination=='median':
        combined=np.median(absn,0)
        argsort=absn.argsort(0)
        combinationMatrix=np.zeros((combined.shape[1],n,combined.shape[0]))
        if n%2==0:
            index1=argsort[int(n/2)-1,:,:]
            index2=argsort[int(n/2),:,:]
            for t in range(combined.shape[0]):
                for i in range(len(index1[t])):
                    combinationMatrix[i,index1[t][i],t]=0.5
                    combinationMatrix[i,index2[t][i],t]=0.5
        else:
            index=argsort[int(n/2),:,:]
            for t in range(combined.shape[0]):
                for i in range(len(index[t])):
                    combinationMatrix[i,index[t][i],t]=1
        extraOut['combinationMatrix']=combinationMatrix
    elif combination=='accuracy':
        combined, accuracy=combinationAccuracy(absn, dfData, **dictCombination)
        extraOut['combinationMatrix']=accuracy
    elif combination=='inverse loss':
        combined, weights=combinationInverseLoss(absn, dfData, **dictCombination)
        extraOut['combinationMatrix']=weights
    else: raise Exception('Combination not recognised.')
    
    return combined, extraOut

def ABSOnePlot(models, df, locParameters, dfData, combinations=None, axes=None, fig=None,
               lossType='Loss k', cmapStyle='magma', linestyles=['-', '--', ':', '-.'],
               colors=['black', 'black', 'black', 'black'], net=mod.FeedForwardLossLogSigma,
               alpha=0.5, title=None, linewidth=1, labels=[None, None, None, None],
               filt=[[-np.inf,np.inf], [-np.inf,np.inf], [-np.inf,np.inf]]):
    """Plots the ABS curves for models in single figures. Also add the combinations
    as given by the list of ABS matrices in combinatinos. """
    if axes is None:
        fig,axes=plt.subplots(1,3)
        axes=axes.flat
    cmap=mpl.cm.get_cmap(cmapStyle)
    for i,m in enumerate(models):
        color=cmap(i/len(models))
        absHat=ABSCurves(m, df, locParameters, dfData, axList=axes, title='', net=net, color=color, alpha=alpha, filt=filt)
    dates=dfData.index
    if combinations is not None:
        for j,comb in enumerate(combinations):
            for i in range(comb.shape[1]):
                axes[i].plot(dates, comb[:,i], color=colors[j], linewidth=linewidth,
                             linestyle=linestyles[j], zorder=2.5, label=labels[j])
    for ax in axes:
        ax.legend()
    fig.autofmt_xdate()
    return fig, axes

def OLS(x, y, CI=False):
    """Performs OLS of 1~x on y, samples in rows. Returns
    estimated beta and sigma squared."""
    X=np.hstack((np.full((len(x),1),1), x))
    n,k=x.shape
    beta=np.linalg.inv(X.T@X)@X.T@y
    eps=y-X@beta
    sigma2=eps.T@eps/(n-k)
    if CI:
        interval=[beta-1.95*np.sqrt(np.diag(np.linalg.inv(X.T@X)*sigma2)).reshape((-1,1)),beta+1.95*np.sqrt(np.diag(np.linalg.inv(X.T@X)*sigma2)).reshape((-1,1))]
        return beta, sigma2, interval
    else:
        return beta, sigma2
    
def JB(x):
    """Returns the Jarque-Bera test statistic."""
    skew=stats.skew(x,None)
    kurt=stats.kurtosis(x,None,False)
    return (len(x)/6)*(skew**2+0.25*(kurt-3)**2)
           
def main():
    pass

if __name__=="__main__":
    __spec__ = None
    pathos.helpers.freeze_support()
    main()
