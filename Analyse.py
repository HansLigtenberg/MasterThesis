"""
Analyse.py

Purpose: Analyse the estimates

Version:
    1   Import AnalyseFunctions and start analysis based on Test8.py
        Switch to ReadData3
    2   Add figures from preliminary experiment
    3   Get results for meeting
    4   
    5   Change figures to fit for first draft
    6   Include Shapley value importance, change dfData to dfDataRec and dfData
    
Author: Hans Ligtenberg
"""
import numpy as np
import Modules as mod
import matplotlib.pyplot as plt
import matplotlib as mpl
import ReadData2
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
import AnalyseFunctions as ana

def main():
###Magic numbers  
    #Normal distribution
    locFinalLoss=os.getcwd()+'/Results/Estimates/Final_loss/Final_loss_'
    locFinalParameters=os.getcwd()+'/Results/Estimates/Final_parameters/Final_parameters_'
    locK=os.getcwd()+'/Results/Estimates/k/k_'
    locP=os.getcwd()+'/Results/Estimates/p/p_'
    locLosses=os.getcwd()+'/Results/Estimates/Losses/Losses_'
    locLossK=os.getcwd()+'/Results/Estimates/Loss_k/Loss_k_'
    locHyperParameters=os.getcwd()+'/Results/Estimates/Hyper-Parameters'
    locMessages=os.getcwd()+'/Results/Estimates/Messages'
    locShapley=os.getcwd()+'/Results/Estimates/Shapley'
    
    #Preliminary experiment
    locFinalLossPre=os.getcwd()+'/Results_preliminary/Estimates/Final_loss/Final_loss_'
    locFinalParametersPre=os.getcwd()+'/Results_preliminary/Estimates/Final_parameters/Final_parameters_'
    locKPre=os.getcwd()+'/Results_preliminary/Estimates/k/k_'
    locPPre=os.getcwd()+'/Results_preliminary/Estimates/p/p_'
    locLossesPre=os.getcwd()+'/Results_preliminary/Estimates/Losses/Losses_'
    locLossKPre=os.getcwd()+'/Results_preliminary/Estimates/Loss_k/Loss_k_'
    locHyperParametersPre=os.getcwd()+'/Results_preliminary/Estimates/Hyper-Parameters'
    locMessagesPre=os.getcwd()+'/Results_preliminary/Estimates/Messages'
    locShapleyPre=os.getcwd()+'/Results_preliminary/Estimates/Shapley'
    
    locStyle=os.getcwd()+'/Plot_styles/Style.mplstyle'
    locTable=os.getcwd()+'/Results/Tables'
    locFigure=os.getcwd()+'/Results/Figures'
    
#Saving figures
    backend='pgf'
    textwidth=452.9679 #In points
    textwidthIn=textwidth/72.27
    textheight=700.5068 #In points
    textheightIn=textheight/72.27
    
###Initialisations
    plt.style.use(locStyle)
    plt.close('all')
    rng=np.random.default_rng(19980528)
    
###Load data
    dfDataRec=ana.prepareData()
    dates=dfDataRec.index
    dfData=dfDataRec[dfDataRec.columns[:-1]]
    
    #Normal distribution
    df, dfTrainVal=ana.results()
    print(df.sort_values('Loss k'))
    
    #Preliminary experiment
    dfPre, dfTrainValPre=ana.results(locHyperParametersPre, locMessagesPre,locFinalLossPre, locKPre, locLossKPre)
    
###Determine combination
    #Adding mean
    dictModel={'addCombination':'mean', 'lossType':'Loss k','lossFunction':ana.lossNormal,
                'net':mod.FeedForwardLossLogSigma, 'p':50, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    dictCombination={'lossFunction':ana.lossNormal, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    modelAverageMeanAdd, extraOut=ana.modelAveraging('add', df, locP, dfData, 'mean', net=mod.FeedForwardLossLogSigma, dictModel=dictModel)
    modelsMeanAdd=extraOut['models']
    combinationMatrixMeanAdd=extraOut['combinationMatrix']
    table=[['Add', 'Mean', modelsMeanAdd, ana.lossNormal(modelAverageMeanAdd, R=dfData['HML'], factor=dfData['M'])]]
    
    #Adding median
    dictModel={'addCombination':'median', 'lossType':'Loss k','lossFunction':ana.lossNormal,
                'net':mod.FeedForwardLossLogSigma, 'p':50, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    dictCombination={'lossFunction':ana.lossNormal, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    modelAverageMedianAdd, extraOut=ana.modelAveraging('add', df, locP, dfData, 'median', net=mod.FeedForwardLossLogSigma, dictModel=dictModel)
    modelsMedianAdd=extraOut['models']
    combinationMatrixMedianAdd=extraOut['combinationMatrix']
    table.append(['Add', 'Median', modelsMedianAdd, ana.lossNormal(modelAverageMedianAdd, R=dfData['HML'], factor=dfData['M'])])
    
    #Adding accuracy
    dictModel={'addCombination':'accuracy', 'lossType':'Loss k','lossFunction':ana.lossNormal,
                'net':mod.FeedForwardLossLogSigma, 'p':50, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    dictCombination={'lossFunction':ana.lossNormal, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    modelAverageAccuracyAdd, extraOut=ana.modelAveraging('add', df, locP, dfData, 'accuracy',net=mod.FeedForwardLossLogSigma, dictModel=dictModel,
                                                      dictCombination=dictCombination)
    modelsAccuracyAdd=extraOut['models']
    combinationMatrixAccuracyAdd=extraOut['combinationMatrix']
    table.append(['Add', 'Accuracy', modelsAccuracyAdd, ana.lossNormal(modelAverageAccuracyAdd, R=dfData['HML'], factor=dfData['M'])])
    
    #Adding inverse loss
    dictModel={'addCombination':'inverse loss', 'lossType':'Loss k','lossFunction':ana.lossNormal,
                'net':mod.FeedForwardLossLogSigma, 'p':50, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    dictCombination={'lossFunction':ana.lossNormal, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    modelAverageInverseLossAdd, extraOut=ana.modelAveraging('add', df, locP, dfData, 'inverse loss',net=mod.FeedForwardLossLogSigma, dictModel=dictModel,
                                                      dictCombination=dictCombination)
    modelsInverseLossAdd=extraOut['models']
    combinationMatrixInverseLossAdd=extraOut['combinationMatrix']
    table.append(['Add', 'Inverse loss', modelsInverseLossAdd, ana.lossNormal(modelAverageInverseLossAdd, R=dfData['HML'], factor=dfData['M'])])
    
    #Median 50
    number=50
    modelAverageMedian50, extraOut=ana.modelAveraging(list(df.nsmallest(50, 'Loss k').index), df, locP, dfData,
                                                        'median', net=mod.FeedForwardLossLogSigma)
    modelsMedian50=extraOut['models']
    combinationMatrixMedian50=extraOut['combinationMatrix']
    table.append(['Best 50', 'Median', 'Best 50', ana.lossNormal(modelAverageMedian50, R=dfData['HML'], factor=dfData['M'])])
    
    dfCombined=pd.DataFrame(table, columns=['Selection', 'Combination', 'Selected estimates', 'Validation loss'])
    print(dfCombined)

    
###Tables and printed output
    variables=['Model','Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Learning rate', 'Momentum', 'Rho', 'k', 'Loss k']
    header=['Estimate','Layer 1', '2', '3', '4', '5', 'Learning rate', 'Mom.', '$\\rho$', 'Updates', 'Val. loss']
    fmt=[ana.intfmt, ana.intfmt, ana.intfmt, ana.intfmt, ana.intfmt, ana.intfmt, ana.nonfmt, ana.decfmt, ana.decfmt, ana.intfmt, ana.decfmt]
    
    variablesFull=['Model','Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5',
                   'Learning rate', 'Momentum', 'Rho', 'k', 'Loss k','Loss']
    headerFull=['Est.','Layer 1', '2', '3', '4', '5', 'Learning rate', 'Mom.',
                '$\\rho$', 'Updates', 'Val. loss', 'Fin. loss']
    fmtFull=[ana.intfmt, ana.intfmt, ana.intfmt, ana.intfmt, ana.intfmt, ana.intfmt,
             ana.nonfmt, ana.decfmt, ana.decfmt, ana.intfmt, ana.decfmt, ana.decfmt]
    
    
    #Best models 
    number=10
    R=dfData['HML'].to_numpy().reshape((-1,1))
    factor=dfData['M'].to_numpy().reshape((-1,1))
    (alphaOLS, betaOLS),sigmaOLS=ana.OLS(factor,R)
    alphaOLS=np.resize(alphaOLS,(len(R),1))
    betaOLS=np.resize(betaOLS,(len(R),1))
    sigmaOLS=np.resize(sigmaOLS,(len(R),1))
    lossOLS=ana.lossNormal(alphaOLS,betaOLS,sigmaOLS,R,factor)
    
    dfBest=df.nsmallest(number,'Loss k').reset_index()[variables]
    dfBest.loc[len(dfBest.index)]=['OLS', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, lossOLS[0]]
    ana.table(dfBest, 'Best_results', locTable,
              column_format='lrrrrrlrrrr', formatters=fmt, longtable=False, header=header, tex=True)
    
    #Full results
    ana.table(df.reset_index()[variablesFull], 'Full_results', locTable,
              column_format='lrrrrrlrrrrr', formatters=fmtFull, longtable=True, header=headerFull, tex=True)
    
    #Data correlation
    ana.table(dfDataRec.corr(), 'Data_correlation', locTable, index=True, tex=True, float_format="%.3f")
    
    #Full results preliminary
    ana.table(dfPre.reset_index()[variablesFull], 'Results_preliminary', locTable, 
              column_format='lrrrrrlrrrrr', formatters=fmtFull, longtable=True, header=headerFull, tex=True)

    #Combined models
    dfCombined.loc[len(dfCombined)]=['OLS', np.nan, np.nan, lossOLS[0]]
    ana.table(dfCombined, 'Results_combined', locTable, tex=True, index=False, float_format="%.4f")
    
    #Updates
    print('Average number of updates of SGD', df[(df['Method']=='Nesterov')]['k'].mean())
    print('Average number of updates of RMSProp', df[(df['Method']=='RMSProp')]['k'].mean())
    print('Average number of updates of SGD with high learning rate', df[(df['Learning rate']=='Lin(0.01,0.001,250)')|(df['Learning rate']=='Lin(0.01,0.001,1000)')]['k'].mean())
    print('Average number of updates of SGD with low learning rate', df[(df['Learning rate']=='Lin(0.001,0.0001,250)')|(df['Learning rate']=='Lin(0.001,0.0001,100)')]['k'].mean())

    # #Correlation
    number=3
    models=list(df.nsmallest(number, 'Loss k').index)
    rec=dfDataRec['REC'].to_numpy().reshape((-1,1))
    vol=dfDataRec['VOL'].to_numpy().reshape((-1,1))
    abs23=ana.ABShat(models[0], df, locP, dfData)
    abs77=ana.ABShat(models[1], df, locP, dfData)
    abs37=ana.ABShat(models[2], df, locP, dfData)
    
    print(f'Average of alpha, beta and sigma for {models[0]} {np.mean(abs23,0)}')
    print(f'Average of alpha, beta and sigma for {models[1]} {np.mean(abs77,0)}')
    print(f'Average of alpha, beta and sigma for {models[2]} {np.mean(abs37,0)}')
    print(f'Average of alpha, beta and sigma for mean {np.mean(modelAverageMeanAdd,0)}')
    print(f'Average of alpha, beta and sigma for median {np.mean(modelAverageMedianAdd,0)}')
    
    print(f'Correlation matrix of alpha, beta, sigma, recession, volatility for {models[0]}')
    print(np.corrcoef(np.hstack((abs23,rec,vol)),rowvar=False))
    print(f'Correlation matrix of alpha, beta, sigma, recession, volatility for {models[1]}')
    print(np.corrcoef(np.hstack((abs77,rec,vol)),rowvar=False))
    print(f'Correlation matrix of alpha, beta, sigma, recession, volatility for {models[2]}')
    print(np.corrcoef(np.hstack((abs37,rec,vol)),rowvar=False))
    print('Correlation matrix of alpha, beta, sigma, recession, volatility for mean')
    print(np.corrcoef(np.hstack((modelAverageMeanAdd,rec,vol)),rowvar=False))
    print('Correlation matrix of alpha, beta, sigma, recession, volatility for median')
    print(np.corrcoef(np.hstack((modelAverageMedianAdd,rec,vol)),rowvar=False))
    
    print(f'Covariance matrix of alpha, beta, sigma, recession, volatility for {models[0]}')
    print(np.cov(np.hstack((abs23,rec,vol)),rowvar=False))
    print(f'Covariance matrix of alpha, beta, sigma, recession, volatilit for {models[1]}')
    print(np.cov(np.hstack((abs77,rec,vol)),rowvar=False))
    print(f'Covariance matrix of alpha, beta, sigma, recession, volatilit for {models[2]}')
    print(np.cov(np.hstack((abs37,rec,vol)),rowvar=False))
    print('Covariance matrix of alpha, beta, sigma, recession, volatility for mean')
    print(np.cov(np.hstack((modelAverageMeanAdd,rec,vol)),rowvar=False))
    print('Covariance matrix of alpha, beta, sigma, recession, volatility for median')
    print(np.cov(np.hstack((modelAverageMedianAdd,rec,vol)),rowvar=False))
    
    #Persistence of beta
    (beta0,phi),sigma2, interval=ana.OLS(abs23[:-1,1].reshape((-1,1)),abs23[1:,1].reshape((-1,1)),True)
    JBalpha=ana.JB(abs23[:,0])
    JBbeta=ana.JB(abs23[:,1])
    pAlpha=1-stats.chi2.cdf(JBalpha,2)
    pBeta=1-stats.chi2.cdf(JBbeta,2)
    print('Persistency of beta: beta0, phi, sigma2', beta0, phi, sigma2)
    print('Confidence interval for beta', interval[0][1], interval[1][1])
    print('Jarque-Bera statistics for alpha and beta and p-values', JBalpha, JBbeta, pAlpha, pBeta)

###Figures
    #Series
    fig, axes=plt.subplots(5,2)
    ana.addRecessions(axes[0,0], dates)
    axes[0,0].plot(dates, dfData['HML'],label='HML', alpha=0.7)
    axes[0,0].plot(dates, dfData['M'], label='M', alpha=0.7)                
    axes[0,0].legend()
    ana.plotStandardised(dfData, ['HML', 'M'], axList=axes.flat[1:])
    ana.saveFigure(fig, 'Series', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1, subplots=(5,2), textheightIn=textheightIn, height=0.92*textheightIn)

    #Loss curves
    number=10
    filt=np.inf
    models=df.nsmallest(number, 'Loss k').index
    fig, axes=plt.subplots(5,2)
    for i,ax in zip(models, axes.flat):
        ana.lossCurve(i, locLosses, df['k'], df['Loss k'], ax=ax, filt=filt)
    for ax in axes[:,0]:
        ax.set(ylabel='Loss')
    for ax in axes[-1,:]:
        ax.set(xlabel='Update')
    ana.saveFigure(fig, 'Loss_curve', locFigure=locFigure, backend=backend, width=textwidthIn,
                    fraction=1, subplots=(5,2), textheightIn=textheightIn*0.97)

    #Histograms
    fig, axes=plt.subplots(1,3)
    axes=axes.flat
    
    #Depth
    number=20
    bins=5
    df1=df['Loss k'][df['Layer 2'].isnull()].nsmallest(number)
    df2=df['Loss k'][(~df['Layer 2'].isnull())&(df['Layer 3'].isnull())].nsmallest(number)
    df3=df['Loss k'][(~df['Layer 3'].isnull())&(df['Layer 5'].isnull())].nsmallest(number)
    df5=df['Loss k'][(~df['Layer 5'].isnull())].nsmallest(number)
    ana.histogram(df1, axes[0], density=True, drawDensity=True, mean=True, bins=bins,
                  label='1 layer', alpha=0.3, linewidth=0.5)
    ana.histogram(df2, axes[0], density=True, drawDensity=True, mean=True, bins=bins,
                  label='2 layers', alpha=0.3, linewidth=0.5)
    ana.histogram(df3, axes[0], density=True, drawDensity=True, mean=True, bins=bins,
                  label='3 layers', alpha=0.3, linewidth=0.5)
    ana.histogram(df5, axes[0], density=True, drawDensity=True, mean=True, bins=bins,
                  label='5 layers', alpha=0.3, linewidth=0.5, xlabel='Loss')
    axes[0].legend()
    
    #Nesterov RMSProp
    number=20
    bins=5
    dfFiltered=df['Loss k'][(df['Method']=='Nesterov')].nsmallest(number)
    ana.histogram(dfFiltered, axes[1], density=True, drawDensity=True, mean=True, bins=bins, label='SGD', xlabel='Loss')
    dfFiltered=df['Loss k'][(df['Method']=='RMSProp')].nsmallest(number)
    ana.histogram(dfFiltered, axes[1], density=True, drawDensity=True, mean=True, bins=bins, label='RMSProp', xlabel='Loss')
    axes[1].legend()
    
    #High number of updates low number of updates
    number=20
    bins=5
    dfFiltered=df['Loss k'][(df['k']>=5000)].nsmallest(number)
    ana.histogram(dfFiltered, axes[2], density=True, drawDensity=True, mean=True, bins=bins, label=r'$\geq 5000$ updates', xlabel='Loss')
    dfFiltered=df['Loss k'][(df['k']<5000)].nsmallest(number)
    ana.histogram(dfFiltered, axes[2], density=True, drawDensity=True, mean=True, bins=bins, label=r'$<5000$ updates', xlabel='Loss')
    axes[2].legend()

    ana.saveFigure(fig, 'Histogram', locFigure, width=textwidthIn, fraction=1, height=0.25*textheightIn)
    
    #Histograms preliminary
    #Depth preliminary
    fig,axes=plt.subplots(1,3)
    number=50
    bins=10
    df1=dfPre['Loss k'][dfPre['Layer 2'].isnull()].nsmallest(number)
    df2=dfPre['Loss k'][(~dfPre['Layer 2'].isnull())&(dfPre['Layer 3'].isnull())].nsmallest(number)
    df3=dfPre['Loss k'][(~dfPre['Layer 3'].isnull())&(dfPre['Layer 5'].isnull())].nsmallest(number)
    df5=dfPre['Loss k'][(~dfPre['Layer 5'].isnull())].nsmallest(number)
    ana.histogram(df1, axes[0], density=True, drawDensity=True, mean=True, bins=bins,
                  label='1', alpha=0.3, linewidth=0.5)
    ana.histogram(df2, axes[0], density=True, drawDensity=True, mean=True, bins=bins,
                  label='2', alpha=0.3, linewidth=0.5)
    ana.histogram(df3, axes[0], density=True, drawDensity=True, mean=True, bins=bins,
                  label='3', alpha=0.3, linewidth=0.5)
    ana.histogram(df5, axes[0], density=True, drawDensity=True, mean=True, bins=bins,
                  label='5', alpha=0.3, linewidth=0.5, xlabel='Loss')
    axes[0].legend()
    
    #Constant width preliminary
    number=50    
    dfConstant=ana.selectArch(dfPre, [[9,9,9], [9,9,9,9]])
    dfDecreasing=ana.selectArch(dfPre, [[9,9,3], [9,9,6,3]])
    dfIncreasing=ana.selectArch(dfPre, [[9,3,9], [9,3,6,9]])
    ana.histogram(dfConstant['Loss k'].nsmallest(number), axes[1], density=True, drawDensity=True,
                  mean=True, bins=10, label='Constant', xlabel='Loss', alpha=0.3)
    ana.histogram(dfIncreasing['Loss k'].nsmallest(number), axes[1], density=True, drawDensity=True,
                  mean=True, bins=10, label='Increasing', xlabel='Loss', alpha=0.3)
    ana.histogram(dfDecreasing['Loss k'].nsmallest(number), axes[1], density=True, drawDensity=True,
                  mean=True, bins=10, label='Decreasing', xlabel='Loss', alpha=0.3)
    axes[1].legend()
   
    #Width preliminary
    number=50
    dfNarrow=ana.selectArch(dfPre, [[9,5,5], [9,4,4,4], [9,3,3,3,3,3]])
    dfWide=ana.selectArch(dfPre, [[9,9,9], [9,9,9,9], [9,9,9,9,9,9]])
    ana.histogram(dfNarrow['Loss k'].nsmallest(number), axes[2], density=True, drawDensity=True, mean=True, bins=10, label='Narrow', xlabel='Loss')
    ana.histogram(dfWide['Loss k'].nsmallest(number), axes[2], density=True, drawDensity=True, mean=True, bins=10, label='Nine', xlabel='Loss')
    axes[2].legend()
    ana.saveFigure(fig, 'Architecture_preliminary', locFigure, width=textwidthIn, fraction=1, height=0.25*textheightIn)
    
    #Nesterov RMSProp preliminary
    fig,axes=plt.subplots(1,3)
    number=50
    dfFiltered=dfPre['Loss k'][(dfPre['Method']=='Nesterov')].nsmallest(number)
    ana.histogram(dfFiltered, axes[0], density=True, drawDensity=True, mean=True, bins=10, label='SGD', xlabel='Loss')
    dfFiltered=dfPre['Loss k'][(dfPre['Method']=='RMSProp')].nsmallest(number)
    ana.histogram(dfFiltered, axes[0], density=True, drawDensity=True, mean=True, bins=10, label='RMSProp', xlabel='Loss')
    axes[0].legend()
    
    #Rho preliminary
    number=50
    dfFiltered=dfPre['Loss k'][(dfPre['Rho']==0.5)].nsmallest(number)
    ana.histogram(dfFiltered, axes[1], density=True, drawDensity=True, mean=True, bins=10, label=r'$\rho$=0.5', xlabel='Loss')
    dfFiltered=dfPre['Loss k'][(dfPre['Rho']==0.9)].nsmallest(number)
    ana.histogram(dfFiltered, axes[1], density=True, drawDensity=True, mean=True, bins=10, label=r'$\rho$=0.9', xlabel='Loss')
    axes[1].legend()
    
    #Momentum preliminary
    number=50
    dfFiltered=dfPre['Loss k'][(dfPre['Momentum']==0)].nsmallest(number)
    ana.histogram(dfFiltered, axes[2], density=True, drawDensity=True, mean=True, bins=10,
                  label='Momentum=0', xlabel='Loss', alpha=0.3)
    dfFiltered=dfPre['Loss k'][(dfPre['Momentum']==0.5)].nsmallest(number)
    ana.histogram(dfFiltered, axes[2], density=True, drawDensity=True, mean=True, bins=10,
                  label='Momentum=0.5', xlabel='Loss', alpha=0.3)
    dfFiltered=dfPre['Loss k'][(dfPre['Momentum']==0.9)].nsmallest(number)
    ana.histogram(dfFiltered, axes[2], density=True, drawDensity=True, mean=True, bins=10,
                  label='Momentum=0.9', xlabel='Loss', alpha=0.3)
    axes[2].legend()
    ana.saveFigure(fig, 'Optimisation_preliminary', locFigure, width=textwidthIn, fraction=1, height=0.25*textheightIn)
   
    #ABS
    number=1
    models=list(df.nsmallest(number, 'Loss k').index)+['Mean', 'Median']
    absMatrices=number*[None]+[modelAverageMeanAdd, modelAverageMedianAdd]
    fig,axes=plt.subplots(len(models),3)
    for j,(i,absMatrix) in enumerate(zip(models,absMatrices)):
        absHat=ana.ABSCurves(i, df, locP, dfData, axList=axes.flat[j*3:],absMatrix=absMatrix)
    fig.autofmt_xdate()
    ana.saveFigure(fig, 'ABS', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1, height=textheightIn*0.5)
    
    #AB
    number=1
    models=list(df.nsmallest(number, 'Loss k').index)+['Mean', 'Median']
    absMatrices=number*[None]+[modelAverageMeanAdd, modelAverageMedianAdd]
    fig,axes=plt.subplots(len(models),2)
    for j,(i,absMatrix) in enumerate(zip(models,absMatrices)):
        absHat=ana.ABCurves(i, df, locP, dfData, axList=axes.flat[j*2:],absMatrix=absMatrix)
    fig.autofmt_xdate()
    ana.saveFigure(fig, 'AB', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1, subplots=(3,2))

    #ABS one plot
    filt=[[-2.5,2.5],[-1.5,2],[0,10]]
    fig1,ax1=plt.subplots()
    fig2,ax2=plt.subplots()
    fig3,ax3=plt.subplots()
    axes=[ax1, ax2, ax3]
    combinations=[abs23]
    number=20
    models=df.nsmallest(number, 'Loss k').index
    ana.ABSOnePlot(models[1:], df, locP, dfData, combinations=combinations, axes=axes, fig=fig1,
                    lossType='Loss k', cmapStyle='magma', linestyles=['-', '--', ':', '-.'],
                    colors=['black', 'black', 'black', 'black'], net=mod.FeedForwardLossLogSigma,
                    alpha=0.5, title=None, linewidth=1, labels=['23', None, None, None, None],
                    filt=filt)
    fig1.autofmt_xdate()
    fig2.autofmt_xdate()
    fig3.autofmt_xdate()
    ana.saveFigure(fig1, 'Alpha', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    ana.saveFigure(fig2, 'Beta', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    ana.saveFigure(fig3, 'Sigma', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)


    #Contribution plot multiple models
    # number=1
    # models=list(df.nsmallest(number, 'Loss k').index)
    # models=models+[modelsMeanAdd,modelsMedianAdd]
    # combinationMatrices=number*[None]+[combinationMatrixMeanAdd,combinationMatrixMedianAdd]
    # modelNames=number*[None]+['Mean', 'Median']
    
    # figList=[]
    # axesList=[]
    # for i in range(9):
    #     fig,axes=plt.subplots(3,len(models))
    #     figList.append(fig)
    #     axesList.append(axes)
    # for i,(m,comb,name) in enumerate(zip(models,combinationMatrices,modelNames)):
    #     axVec=[]
    #     for j in range(3):
    #         for k in range(3):
    #             for l in range(3):
    #                 axVec.append(axesList[j*3+l][k,i])
    #     ana.contributionPlot(m, df, locP, dfData, steps=100, axList=axVec, combinationMatrix=comb, modelName=name)
    # handles, labels=axes[0,0].get_legend_handles_labels()
    # for i in range(3):
        
    #     leg=figList[i*3].legend(handles, labels, loc='center right')
    #     ana.saveFigure(figList[i*3], f'Contribution_alpha_{i}', locFigure=locFigure, backend=backend,
    #                     width=textwidthIn, height=0.8*textheightIn, fraction=1, textheightIn=textheightIn)
    #     leg=figList[i*3+1].legend(handles, labels, loc='center right')
    #     ana.saveFigure(figList[i*3+1], f'Contribution_beta_{i}', locFigure=locFigure, backend=backend,
    #                     width=textwidthIn, height=0.8*textheightIn, fraction=1, textheightIn=textheightIn)
    #     leg=figList[i*3+2].legend(handles, labels, loc='center right')
    #     ana.saveFigure(figList[i*3+2], f'Contribution_sigma_{i}', locFigure=locFigure, backend=backend,
    #                     width=textwidthIn, height=0.8*textheightIn, fraction=1, textheightIn=textheightIn)
    
    #Contribution plot one model, one parameter
    number=1
    models=list(df.nsmallest(number, 'Loss k').index)
    models=models+[modelsMeanAdd,modelsMedianAdd]
    combinationMatrices=number*[None]+[combinationMatrixMeanAdd,combinationMatrixMedianAdd]
    modelNames=number*[None]+['Mean', 'Median']
    for m,c,n in zip(models,combinationMatrices,modelNames):
        figAlpha,axesAlpha=plt.subplots(3,3)
        figBeta,axesBeta=plt.subplots(3,3)
        figSigma,axesSigma=plt.subplots(3,3)
        axList=[]
        for a,b,s in zip(axesAlpha.flat,axesBeta.flat,axesSigma.flat):
            axList=axList+[a,b,s]
        ana.contributionPlot(m, df, locP, dfData, steps=100, axList=axList, combinationMatrix=c, modelName=n)
        n=str(m) if n is None else n
        handles, labels=a.get_legend_handles_labels()
        leg=figAlpha.legend(handles, labels, loc='center right')
        ana.saveFigure(figAlpha, f'Contribution_alpha_{n}', locFigure=locFigure, backend=backend,
                        width=textwidthIn, height=0.8*textheightIn, fraction=1, textheightIn=textheightIn)
        leg=figBeta.legend(handles, labels, loc='center right')
        ana.saveFigure(figBeta, f'Contribution_beta_{n}', locFigure=locFigure, backend=backend,
                        width=textwidthIn, height=0.8*textheightIn, fraction=1, textheightIn=textheightIn)
        leg=figSigma.legend(handles, labels, loc='center right')
        ana.saveFigure(figSigma, f'Contribution_sigma_{n}', locFigure=locFigure, backend=backend,
                        width=textwidthIn, height=0.8*textheightIn, fraction=1, textheightIn=textheightIn)
      
    #Heat maps
    number=10
    models=list(df.nsmallest(number, 'Loss k').index)+[modelsMeanAdd, modelsMedianAdd]
    combinationMatrix=number*[None]+[combinationMatrixMeanAdd, combinationMatrixMedianAdd]
    
    #Variable importance
    dfImportanceVI=ana.importanceDf(models, df, locP, dfTrainVal, method='importance', inputName='input',
                                  locShapley=locShapley, relative=True, draws=10,
                                  rng=rng, outputSSD='loss', outputName='output', sample='whole', 
                                  X=['DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP'], 
                                  net=mod.FeedForwardLossLogSigma, combinationMatrix=combinationMatrix,
                                  lossFunction=ana.lossNormal, factor='M', R='HML', dfData=dfData,
                                  combinationNames=['Mean', 'Median'])
    dfImportanceStdVI=ana.standardise(dfImportanceVI)
    fig, ax=plt.subplots()
    ana.heatmap(dfImportanceStdVI, ax=ax)
    ana.saveFigure(fig, f'Importance_standardised_{number}', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    ana.table(dfImportanceStdVI.reset_index(), f'Importance_standardised_{number}', locTable, longtable=True, float_format="%.3f")
    
    #SSD
    dfImportanceSSD=ana.importanceDf(models, df, locP, dfTrainVal, method='SSD', inputName='input',
                                  locShapley=locShapley, relative=True, draws=10,
                                  rng=rng, outputSSD='loss', outputName='output', sample='whole', 
                                  X=['DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP'], 
                                  net=mod.FeedForwardLossLogSigma, combinationMatrix=combinationMatrix,
                                  lossFunction=ana.lossNormal, factor='M', R='HML', dfData=dfData,
                                  combinationNames=['Mean', 'Median'])
    dfImportanceStdSSD=ana.standardise(dfImportanceSSD)
    fig, ax=plt.subplots()
    ana.heatmap(dfImportanceStdSSD, ax=ax)
    ana.saveFigure(fig, f'SSD_loss_standardised_{number}', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    ana.table(dfImportanceStdSSD.reset_index(), f'SSD_loss_standardised_{number}', locTable, longtable=False, float_format="%.3f")
    
    fig,ax=ana.heatmapMultiple([dfImportanceStdVI,dfImportanceStdSSD], xlabels=None, ylabels=None, cbarlabel='Relative importance',
                    variableNames=None, models=None, titles=['VI', 'SSD'])
    ana.saveFigure(fig, 'Heatmaps', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    
    #Shapley value importance
    draws=50
    dfImportanceShapley=ana.importanceDf(models, df, locP, dfTrainVal, method='Shapley', inputName='input',
                                  locShapley=locShapley, relative=True, draws=draws,
                                  rng=rng, outputSSD='loss', outputName='output', sample='whole', 
                                  X=['DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP'], 
                                  net=mod.FeedForwardLossLogSigma, combinationMatrix=combinationMatrix,
                                  lossFunction=ana.lossNormal, factor='M', R='HML', dfData=dfData,
                                  combinationNames=['Mean', 'Median'])
    dfImportanceStdShapley=ana.standardise(dfImportanceShapley)
    fig, ax=plt.subplots()
    ana.heatmap(dfImportanceStdShapley, ax=ax)
    ana.saveFigure(fig, f'Shapley_loss_standardised_{number}_{draws}', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    ana.table(dfImportanceStdSSD.reset_index(), f'Shapley_loss_standardised_{number}_{draws}', locTable, longtable=False, float_format="%.3f")
    
    fig,ax=ana.heatmapMultiple([dfImportanceStdVI,dfImportanceStdSSD,dfImportanceStdShapley], xlabels=None, ylabels=None, cbarlabel='Relative importance',
                    variableNames=None, models=None, titles=['VI', 'SSD', 'SVI'])
    ana.saveFigure(fig, f'Heatmaps_{draws}', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)

    #SSD ABS
    dfImportanceSSDABS=ana.importanceDf(models, df, locP, dfTrainVal, method='SSD', inputName='input',
                                  locShapley=locShapley, relative=True, draws=10,
                                  rng=rng, outputSSD='abs', outputName='output', sample='whole', 
                                  X=['DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP'], 
                                  net=mod.FeedForwardLossLogSigma, combinationMatrix=combinationMatrix,
                                  lossFunction=ana.lossNormal, factor='M', R='HML', dfData=dfData,
                                  combinationNames=['Mean', 'Median'])
    dfImportanceStdSSDABS=ana.standardise(dfImportanceSSDABS)
    fig, axes=ana.ABSNHeatmap2(dfImportanceStdSSDABS)
    ana.saveFigure(fig, f'SSD_ABS_standardised_{number}', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    ana.table(dfImportanceStdSSDABS, f'SSD_ABS_standardised_{number}', locTable, longtable=False, float_format="%.3f", multirow=True, index=True)
    
    fig,ax=ana.heatmapMultiple([dfImportanceStdSSDABS.xs(r'$\alpha$',level=1),dfImportanceStdSSDABS.xs(r'$\beta$',level=1)],
                                xlabels=None, ylabels=None, cbarlabel='Relative importance', variableNames=None, models=None,
                                titles=[r'$\hat\alpha$', r'$\hat\beta$'])
    ana.saveFigure(fig, f'SSD_AB_standardised_{number}', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    
    #Histograms of alpha and beta
    fig,axes=plt.subplots(1,2)
    ana.histogram(abs23[:,0], axes[0], density=True, drawDensity=True, mean=True, label='23', xlabel=r'$\hat\alpha$', alpha=0.7, densityExtend=0.2)
    # ana.histogram(abs77[:,0], axes[0], density=True, drawDensity=True, mean=True, label='77', xlabel=r'$\alpha$', alpha=0.5)
    # ana.histogram(abs37[:,0], axes[0], density=True, drawDensity=True, mean=True, label='37', xlabel=r'$\alpha$', alpha=0.5)
    # ana.histogram(modelAverageMeanAdd[:,0], axes[0], density=True, drawDensity=True, mean=True, label='Mean', xlabel=r'$\alpha$', alpha=0.5)
    # ana.histogram(modelAverageMedianAdd[:,0], axes[0], density=True, drawDensity=True, mean=True, label='Median', xlabel=r'$\alpha$', alpha=0.5)
    # axes[0].legend()
    
    ana.histogram(abs23[:,1], axes[1], density=True, drawDensity=True, mean=True, label='23', xlabel=r'$\hat\beta$', alpha=0.7, densityExtend=0.2)
    # ana.histogram(abs77[:,1], axes[1], density=True, drawDensity=True, mean=True, label='77', xlabel=r'$\beta$', alpha=0.5)
    # ana.histogram(abs37[:,1], axes[1], density=True, drawDensity=True, mean=True, label='37', xlabel=r'$\beta$', alpha=0.5)
    # ana.histogram(modelAverageMeanAdd[:,1], axes[1], density=True, drawDensity=True, mean=True, label='Mean', xlabel=r'$\beta$', alpha=0.5)
    # ana.histogram(modelAverageMedianAdd[:,1], axes[1], density=True, drawDensity=True, mean=True, label='Median', xlabel=r'$\beta$', alpha=0.5)
    # axes[1].legend() 
    ana.saveFigure(fig, 'Alpha_beta_histogram', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1, subplots=(1,2))

if __name__=="__main__":
    __spec__ = None
    pathos.helpers.freeze_support()
    main()
