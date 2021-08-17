"""
Analyse.py

Purpose: Analyse the estimates for the final loss

Version:
    1   Based on Analyse3.py
        Change data
    
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
    textwidth=418.25368 #In points
    textwidthIn=textwidth/72.27
    textheight=591.5302 #In points
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
    print(df.sort_values('Loss'))
    
    
###Determine combination
    #Adding mean
    dictModel={'addCombination':'mean', 'lossType':'Loss','lossFunction':ana.lossNormal,
                'net':mod.FeedForwardLossLogSigma, 'p':50, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    dictCombination={'lossFunction':ana.lossNormal, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    modelAverageMeanAdd, extraOut=ana.modelAveraging('add', df, locFinalParameters, dfData, 'mean', net=mod.FeedForwardLossLogSigma, dictModel=dictModel)
    modelsMeanAdd=extraOut['models']
    combinationMatrixMeanAdd=extraOut['combinationMatrix']
    table=[['Add', 'Mean', modelsMeanAdd, ana.lossNormal(modelAverageMeanAdd, R=dfData['HML'], factor=dfData['M'])]]
    
    #Adding median
    dictModel={'addCombination':'median', 'lossType':'Loss','lossFunction':ana.lossNormal,
                'net':mod.FeedForwardLossLogSigma, 'p':50, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    dictCombination={'lossFunction':ana.lossNormal, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    modelAverageMedianAdd, extraOut=ana.modelAveraging('add', df, locFinalParameters, dfData, 'median', net=mod.FeedForwardLossLogSigma, dictModel=dictModel)
    modelsMedianAdd=extraOut['models']
    combinationMatrixMedianAdd=extraOut['combinationMatrix']
    table.append(['Add', 'Median', modelsMedianAdd, ana.lossNormal(modelAverageMedianAdd, R=dfData['HML'], factor=dfData['M'])])
    
    #Adding accuracy
    dictModel={'addCombination':'accuracy', 'lossType':'Loss','lossFunction':ana.lossNormal,
                'net':mod.FeedForwardLossLogSigma, 'p':50, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    dictCombination={'lossFunction':ana.lossNormal, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    modelAverageAccuracyAdd, extraOut=ana.modelAveraging('add', df, locFinalParameters, dfData, 'accuracy',net=mod.FeedForwardLossLogSigma, dictModel=dictModel,
                                                      dictCombination=dictCombination)
    modelsAccuracyAdd=extraOut['models']
    combinationMatrixAccuracyAdd=extraOut['combinationMatrix']
    table.append(['Add', 'Accuracy', modelsAccuracyAdd, ana.lossNormal(modelAverageAccuracyAdd, R=dfData['HML'], factor=dfData['M'])])
    
    #Adding inverse loss
    dictModel={'addCombination':'inverse loss', 'lossType':'Loss','lossFunction':ana.lossNormal,
                'net':mod.FeedForwardLossLogSigma, 'p':50, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    dictCombination={'lossFunction':ana.lossNormal, 'R':'HML', 'factor':'M', 'fracTrain':0.8, 'sample':'validation'}
    modelAverageInverseLossAdd, extraOut=ana.modelAveraging('add', df, locFinalParameters, dfData, 'inverse loss',net=mod.FeedForwardLossLogSigma, dictModel=dictModel,
                                                      dictCombination=dictCombination)
    modelsInverseLossAdd=extraOut['models']
    combinationMatrixInverseLossAdd=extraOut['combinationMatrix']
    table.append(['Add', 'Inverse loss', modelsInverseLossAdd, ana.lossNormal(modelAverageInverseLossAdd, R=dfData['HML'], factor=dfData['M'])])
    
    #Median 50
    number=50
    modelAverageMedian50, extraOut=ana.modelAveraging(list(df.nsmallest(50, 'Loss').index), df, locFinalParameters, dfData,
                                                        'median', net=mod.FeedForwardLossLogSigma)
    modelsMedian50=extraOut['models']
    combinationMatrixMedian50=extraOut['combinationMatrix']
    table.append(['Best 50', 'Median', 'Best 50', ana.lossNormal(modelAverageMedian50, R=dfData['HML'], factor=dfData['M'])])
    
    dfCombined=pd.DataFrame(table, columns=['Selection', 'Combination', 'Selected estimates', 'Validation loss'])
    print(dfCombined)

    
###Tables and printed output
    variables=['Model','Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Learning rate', 'Momentum', 'Rho', 'k', 'Loss']
    header=['Model','Layer 1', '2', '3', '4', '5', 'Learning rate', 'Mom.', '$\\rho$', 'Updates', 'Ret. val. loss']
    fmt=[ana.intfmt, ana.intfmt, ana.intfmt, ana.intfmt, ana.intfmt, ana.intfmt, ana.nonfmt, ana.decfmt, ana.decfmt, ana.intfmt, ana.decfmt]
    
    variablesFull=['Est.','Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5',
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
    
    dfBest=df.nsmallest(number,'Loss').reset_index()[variables]
    dfBest.loc[len(dfBest.index)]=['OLS', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, lossOLS[0]]
    ana.table(dfBest, 'Best_results_retrained', locTable,
              column_format='lrrrrrlrrrr', formatters=fmt, longtable=True, header=header, tex=True)
    
    #Combined models
    dfCombined.loc[len(dfCombined)]=['OLS', np.nan, np.nan, lossOLS[0]]
    ana.table(dfCombined, 'Results_combined_retrained', locTable, tex=True, index=False, float_format="%.4f")
    
    #Updates
    print('Average number of updates of SGD', df[(df['Method']=='Nesterov')]['k'].mean())
    print('Average number of updates of RMSProp', df[(df['Method']=='RMSProp')]['k'].mean())
    print('Average number of updates of SGD with high learning rate', df[(df['Learning rate']=='Lin(0.01,0.001,250)')|(df['Learning rate']=='Lin(0.01,0.001,1000)')]['k'].mean())
    print('Average number of updates of SGD with low learning rate', df[(df['Learning rate']=='Lin(0.001,0.0001,250)')|(df['Learning rate']=='Lin(0.001,0.0001,100)')]['k'].mean())

    # #Correlation
    number=3
    models=list(df.nsmallest(number, 'Loss').index)
    rec=dfDataRec['REC'].to_numpy().reshape((-1,1))
    vol=dfData['VOL'].to_numpy().reshape((-1,1))
    abs0=ana.ABShat(models[0], df, locFinalParameters, dfData)
    abs1=ana.ABShat(models[1], df, locFinalParameters, dfData)
    abs2=ana.ABShat(models[2], df, locFinalParameters, dfData)
    
    print(f'Average of alpha, beta and sigma for {models[0]} {np.mean(abs0,0)}')
    print(f'Average of alpha, beta and sigma for {models[1]} {np.mean(abs1,0)}')
    print(f'Average of alpha, beta and sigma for {models[2]} {np.mean(abs2,0)}')
    print(f'Average of alpha, beta and sigma for mean {np.mean(modelAverageMeanAdd,0)}')
    print(f'Average of alpha, beta and sigma for median {np.mean(modelAverageMedianAdd,0)}')
    
    print(f'Correlation matrix of alpha, beta, sigma, recession, volatility for {models[0]}')
    print(np.corrcoef(np.hstack((abs0,rec,vol)),rowvar=False))
    print(f'Correlation matrix of alpha, beta, sigma, recession, volatility for {models[1]}')
    print(np.corrcoef(np.hstack((abs1,rec,vol)),rowvar=False))
    print(f'Correlation matrix of alpha, beta, sigma, recession, volatility for {models[2]}')
    print(np.corrcoef(np.hstack((abs2,rec,vol)),rowvar=False))
    print('Correlation matrix of alpha, beta, sigma, recession, volatility for mean')
    print(np.corrcoef(np.hstack((modelAverageMeanAdd,rec,vol)),rowvar=False))
    print('Correlation matrix of alpha, beta, sigma, recession, volatility for median')
    print(np.corrcoef(np.hstack((modelAverageMedianAdd,rec,vol)),rowvar=False))
    
    print(f'Covariance matrix of alpha, beta, sigma, recession, volatility for {models[0]}')
    print(np.cov(np.hstack((abs0,rec,vol)),rowvar=False))
    print(f'Covariance matrix of alpha, beta, sigma, recession, volatilit for {models[1]}')
    print(np.cov(np.hstack((abs1,rec,vol)),rowvar=False))
    print(f'Covariance matrix of alpha, beta, sigma, recession, volatilit for {models[2]}')
    print(np.cov(np.hstack((abs2,rec,vol)),rowvar=False))
    print('Covariance matrix of alpha, beta, sigma, recession, volatility for mean')
    print(np.cov(np.hstack((modelAverageMeanAdd,rec,vol)),rowvar=False))
    print('Covariance matrix of alpha, beta, sigma, recession, volatility for median')
    print(np.cov(np.hstack((modelAverageMedianAdd,rec,vol)),rowvar=False))
    
    #Persistence of beta
    (beta0,phi),sigma2, interval=ana.OLS(abs0[:-1,1].reshape((-1,1)),abs0[1:,1].reshape((-1,1)),True)
    JBalpha=ana.JB(abs0[:,0])
    JBbeta=ana.JB(abs0[:,1])
    pAlpha=1-stats.chi2.cdf(JBalpha,2)
    pBeta=1-stats.chi2.cdf(JBbeta,2)
    print('Persistency of beta: beta0, phi, sigma2', beta0, phi, sigma2)
    print('Confidence interval for beta', interval[0][1], interval[1][1])
    print('Jarque-Bera statistics for alpha and beta and p-values', JBalpha, JBbeta, pAlpha, pBeta)

###Figures
    #Scatter plot
    fig, ax=plt.subplots()
    dfFiltered=df[(df['Loss k']<5)&(df['Loss']<5)]
    ana.scatter(dfFiltered['Loss k'], dfFiltered['Loss'], ax=ax, xlabel='Loss', ylabel='Retrained loss')
    ana.saveFigure(fig, 'Scatter', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)

    #Histograms
    fig, axes=plt.subplots(1,3)
    axes=axes.flat
    
    #Depth
    number=20
    bins=5
    df1=df['Loss'][df['Layer 2'].isnull()].nsmallest(number)
    df2=df['Loss'][(~df['Layer 2'].isnull())&(df['Layer 3'].isnull())].nsmallest(number)
    df3=df['Loss'][(~df['Layer 3'].isnull())&(df['Layer 5'].isnull())].nsmallest(number)
    df5=df['Loss'][(~df['Layer 5'].isnull())].nsmallest(number)
    ana.histogram(df1, axes[0], density=True, drawDensity=True, mean=True, bins=bins, label='1 layer', alpha=0.5)
    ana.histogram(df2, axes[0], density=True, drawDensity=True, mean=True, bins=bins, label='2 layers', alpha=0.5)
    ana.histogram(df3, axes[0], density=True, drawDensity=True, mean=True, bins=bins, label='3 layers', alpha=0.5)
    ana.histogram(df5, axes[0], density=True, drawDensity=True, mean=True, bins=bins, label='5 layers', alpha=0.5, xlabel='Loss')
    axes[0].legend()
    
    #Nesterov RMSProp
    number=30
    bins=5
    dfFiltered=df['Loss'][(df['Method']=='Nesterov')].nsmallest(number)
    ana.histogram(dfFiltered, axes[1], density=True, drawDensity=True, mean=True, bins=bins, label='SGD', xlabel='Loss')
    dfFiltered=df['Loss'][(df['Method']=='RMSProp')].nsmallest(number)
    ana.histogram(dfFiltered, axes[1], density=True, drawDensity=True, mean=True, bins=bins, label='RMSProp', xlabel='Loss')
    axes[1].legend()
    
    #High number of updates low number of updates
    number=30
    bins=5
    dfFiltered=df['Loss'][(df['k']>=5000)].nsmallest(number)
    ana.histogram(dfFiltered, axes[2], density=True, drawDensity=True, mean=True, bins=bins, label=r'$\geq 5000$ updates', xlabel='Loss')
    dfFiltered=df['Loss'][(df['k']<5000)].nsmallest(number)
    ana.histogram(dfFiltered, axes[2], density=True, drawDensity=True, mean=True, bins=bins, label=r'$<5000$ updates', xlabel='Loss')
    axes[2].legend()

    ana.saveFigure(fig, 'Histogram_retrained', locFigure, width=textwidthIn, fraction=1)
    
    #ABS
    number=1
    models=list(df.nsmallest(number, 'Loss').index)+['Mean', 'Median']
    absMatrices=number*[None]+[modelAverageMeanAdd, modelAverageMedianAdd]
    fig,axes=plt.subplots(len(models),3)
    for j,(i,absMatrix) in enumerate(zip(models,absMatrices)):
        absHat=ana.ABSCurves(i, df, locFinalParameters, dfData, axList=axes.flat[j*3:],absMatrix=absMatrix)
    fig.autofmt_xdate()
    ana.saveFigure(fig, 'ABS_retrained', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1, height=textheightIn*0.85)
    
    #AB
    number=1
    models=list(df.nsmallest(number, 'Loss').index)+['Mean', 'Median']
    absMatrices=number*[None]+[modelAverageMeanAdd, modelAverageMedianAdd]
    fig,axes=plt.subplots(len(models),2)
    for j,(i,absMatrix) in enumerate(zip(models,absMatrices)):
        absHat=ana.ABCurves(i, df, locFinalParameters, dfData, axList=axes.flat[j*2:],absMatrix=absMatrix)
    fig.autofmt_xdate()
    ana.saveFigure(fig, 'AB_retrained', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1, subplots=(3,2))

    #ABS one plot
    filt=[[-2.5,2.5],[-1.5,2],[0,10]]
    fig1,ax1=plt.subplots()
    fig2,ax2=plt.subplots()
    fig3,ax3=plt.subplots()
    axes=[ax1, ax2, ax3]
    combinations=[abs0, modelAverageMeanAdd, modelAverageMedianAdd]
    number=20
    models=df.nsmallest(number, 'Loss').index
    ana.ABSOnePlot(models[1:], df, locFinalParameters, dfData, combinations=combinations, axes=axes, fig=fig1,
                    lossType='Loss', cmapStyle='magma', linestyles=['-', '--', ':', '-.'],
                    colors=['black', 'black', 'black', 'black'], net=mod.FeedForwardLossLogSigma,
                    alpha=0.5, title=None, linewidth=0.5, labels=['23', 'Mean', 'Median', None, None],
                    filt=filt)
    fig1.autofmt_xdate()
    fig2.autofmt_xdate()
    fig3.autofmt_xdate()
    ana.saveFigure(fig1, 'Alpha_retrained', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    ana.saveFigure(fig2, 'Beta_retrained', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    ana.saveFigure(fig3, 'Sigma_retrained', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)


    #Contribution plot multiple models
    number=1
    models=list(df.nsmallest(number, 'Loss').index)
    models=models+[modelsMeanAdd,modelsMedianAdd]
    combinationMatrices=number*[None]+[combinationMatrixMeanAdd,combinationMatrixMedianAdd]
    modelNames=number*[None]+['Mean', 'Median']
    
    figList=[]
    axesList=[]
    for i in range(9):
        fig,axes=plt.subplots(3,len(models))
        figList.append(fig)
        axesList.append(axes)
    for i,(m,comb,name) in enumerate(zip(models,combinationMatrices,modelNames)):
        axVec=[]
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    axVec.append(axesList[j*3+l][k,i])
        ana.contributionPlot(m, df, locFinalParameters, dfData, steps=100, axList=axVec, combinationMatrix=comb, modelName=name)
    handles, labels = axes[0,0].get_legend_handles_labels()
    for i in range(3):
        # plt.rcParams['figure.autolayout']=False
        # plt.rcParams['figure.constrained_layout.use']=True
        
        leg=figList[i*3].legend(handles, labels, loc='center right')
        # leg.set_in_layout(False)
        ana.saveFigure(figList[i*3], f'Contribution_alpha_{i}_retrained', locFigure=locFigure, backend=backend,
                        width=textwidthIn, height=0.8*textheightIn, fraction=1, textheightIn=textheightIn)
        leg=figList[i*3+1].legend(handles, labels, loc='center right')
        # leg.set_in_layout(False)
        ana.saveFigure(figList[i*3+1], f'Contribution_beta_{i}_retrained', locFigure=locFigure, backend=backend,
                        width=textwidthIn, height=0.8*textheightIn, fraction=1, textheightIn=textheightIn)
        leg=figList[i*3+2].legend(handles, labels, loc='center right')
        # leg.set_in_layout(False)
        ana.saveFigure(figList[i*3+2], f'Contribution_sigma_{i}_retrained', locFigure=locFigure, backend=backend,
                        width=textwidthIn, height=0.8*textheightIn, fraction=1, textheightIn=textheightIn)
       
    #Heat maps
    number=10
    models=list(df.nsmallest(number, 'Loss').index)+[modelsMeanAdd, modelsMedianAdd]
    combinationMatrix=number*[None]+[combinationMatrixMeanAdd, combinationMatrixMedianAdd]
    
    #Variable importance
    dfImportance=ana.importanceDf(models, df, locFinalParameters, dfTrainVal, method='importance', inputName='input',
                                  locShapley=locShapley, relative=True, draws=10,
                                  rng=rng, outputSSD='loss', outputName='output', sample='whole', 
                                  X=['DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP'], 
                                  net=mod.FeedForwardLossLogSigma, combinationMatrix=combinationMatrix,
                                  lossFunction=ana.lossNormal, factor='M', R='HML', dfData=dfData,
                                  combinationNames=['Mean', 'Median'])
    dfImportanceStd=ana.standardise(dfImportance)
    fig, ax=plt.subplots()
    ana.heatmap(dfImportanceStd, ax=ax)
    ana.saveFigure(fig, f'Importance_standardised_{number}_retrained', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    ana.table(dfImportance.reset_index(), f'Importance_standardised_{number}_retrained', locTable, longtable=True, float_format="%.3f")
    
    #SSD
    dfImportance=ana.importanceDf(models, df, locFinalParameters, dfTrainVal, method='SSD', inputName='input',
                                  locShapley=locShapley, relative=True, draws=10,
                                  rng=rng, outputSSD='loss', outputName='output', sample='whole', 
                                  X=['DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP'], 
                                  net=mod.FeedForwardLossLogSigma, combinationMatrix=combinationMatrix,
                                  lossFunction=ana.lossNormal, factor='M', R='HML', dfData=dfData,
                                  combinationNames=['Mean', 'Median'])
    dfImportanceStd=ana.standardise(dfImportance)
    fig, ax=plt.subplots()
    ana.heatmap(dfImportanceStd, ax=ax)
    ana.saveFigure(fig, f'SSD_loss_standardised_{number}', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    ana.table(dfImportance.reset_index(), f'SSD_loss_standardised_{number}_retrained', locTable, longtable=True, float_format="%.3f")
    
    #SSD ABS
    dfImportance=ana.importanceDf(models, df, locFinalParameters, dfTrainVal, method='SSD', inputName='input',
                                  locShapley=locShapley, relative=True, draws=10,
                                  rng=rng, outputSSD='abs', outputName='output', sample='whole', 
                                  X=['DEF', 'TERM', 'RREL', 'DP', 'PE', 'VOL', 'INF', 'UE', 'IP'], 
                                  net=mod.FeedForwardLossLogSigma, combinationMatrix=combinationMatrix,
                                  lossFunction=ana.lossNormal, factor='M', R='HML', dfData=dfData,
                                  combinationNames=['Mean', 'Median'])
    dfImportanceStd=ana.standardise(dfImportance)
    fig, axes=ana.ABSNHeatmap2(dfImportanceStd)
    ana.saveFigure(fig, f'SSD_ABS_standardised_{number}_retrained', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1)
    ana.table(dfImportance, f'SSD_ABS_standardised_{number}_retrained', locTable, longtable=True, float_format="%.3f", multirow=True, index=True)
      
    #Historgrams of alpha and beta
    fig,axes=plt.subplots(1,2)
    ana.histogram(abs0[:,0], axes[0], density=True, drawDensity=True, mean=True, label='23', xlabel=r'$\alpha$', alpha=0.5)
    # ana.histogram(abs1[:,0], axes[0], density=True, drawDensity=True, mean=True, label='77', xlabel=r'$\alpha$', alpha=0.5)
    # ana.histogram(abs37[:,0], axes[0], density=True, drawDensity=True, mean=True, label='37', xlabel=r'$\alpha$', alpha=0.5)
    ana.histogram(modelAverageMeanAdd[:,0], axes[0], density=True, drawDensity=True, mean=True, label='Mean', xlabel=r'$\alpha$', alpha=0.5)
    ana.histogram(modelAverageMedianAdd[:,0], axes[0], density=True, drawDensity=True, mean=True, label='Median', xlabel=r'$\alpha$', alpha=0.5)
    axes[0].legend()
    
    ana.histogram(abs0[:,1], axes[1], density=True, drawDensity=True, mean=True, label='23', xlabel=r'$\beta$', alpha=0.5)
    # ana.histogram(abs1[:,1], axes[1], density=True, drawDensity=True, mean=True, label='77', xlabel=r'$\beta$', alpha=0.5)
    # ana.histogram(abs37[:,1], axes[1], density=True, drawDensity=True, mean=True, label='37', xlabel=r'$\beta$', alpha=0.5)
    ana.histogram(modelAverageMeanAdd[:,1], axes[1], density=True, drawDensity=True, mean=True, label='Mean', xlabel=r'$\beta$', alpha=0.5)
    ana.histogram(modelAverageMedianAdd[:,1], axes[1], density=True, drawDensity=True, mean=True, label='Median', xlabel=r'$\beta$', alpha=0.5)
    axes[1].legend() 
    ana.saveFigure(fig, 'Alpha_beta_histogram_retrained', locFigure=locFigure, backend=backend, width=textwidthIn, fraction=1, subplots=(1,2))

if __name__=="__main__":
    __spec__ = None
    pathos.helpers.freeze_support()
    main()
