from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from scipy.stats import ttest_ind as ttest
import scipy.stats as sstats
import numpy
from pandas import *
from pandas import DataFrame as DF
import pickle
from myUtils import *

resultsPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\'
Labels=DF.from_csv(resultsPath+'LabelsAllSubjects.csv')
#import raw Data
rawDF=DF.from_csv(resultsPath+'DATA_AllrawDF.csv').dropna(how='all',axis=0)
##import features data
#rawDF=DF.from_csv(resultsPath+'rawFeaturesDF.csv',header=[0,1],index_col=[0,1])
#rawDF.index.names=['subject','piece_ind']
#rawDF.columns.names=['ftype','signal']


subjectsDetails=DF.from_csv(resultsPath+'SubjectsDetailsDF2-fill with data from michael.csv',index_col=[0])
subjectsDetails.index.names=['subject']

#rescoring PANSS

PANSSindex=subjectsDetails.columns[18:48]
PANSS_elena=subjectsDetails[PANSSindex].copy()
PANSS_michael=DF.from_csv(resultsPath+'RescoringPANSS-Michael.csv')
PANSS_michael=PANSS_michael.loc[PANSSindex].T.copy()

elena_subjects=subjectsDetails.index.copy()
michael_subjects=PANSS_michael.index.copy()
subjectsArrangedList=DF(index=michael_subjects,columns=['elena'])
for ms in michael_subjects:
    for es in elena_subjects:
        if ms[-4:] in es:
            subjectsArrangedList['elena'].loc[ms]=es
            break
PANSS_michael.index=subjectsArrangedList.values.flatten()
PANSS_michael.convert_objects(convert_numeric=True)
toInt=lambda s: int(s) 
PANSS_michael.applymap(toInt) #convert strings to int


PANSSscoringCorr=DF(index=PANSSindex,columns=['n','r','pval'])
R={}
Pval={}
toInt=lambda s: int(s)
for j in PANSSindex:
    pElena=PANSS_elena.sort()[j]
    pMichael=PANSS_michael.sort()[j]
    pMichael=pMichael.dropna(axis=0)
    numvalues=[int(s) for s in pMichael.values.flatten()]
    pMichael=Series(numvalues, index=pMichael.index)
    pElena=pElena.T[pMichael.index].T #only take subjects that were rated for michael
    if len(pMichael):
        PANSSscoringCorr['n']=len(pMichael)
        PANSSscoringCorr['r'].loc[j],PANSSscoringCorr['pval'].loc[j]=sstats.pearsonr(pElena,pMichael)
            



subjectMean=rawDF.mean(axis=0,level='subject').fillna(0)
subjectMeanFeatures=subjectMean.mean(axis=1,level='ftype')

varAll=subjectMean
varAll['ExpressionLength']=varAll['ExpressionLength']/100
featureTypes=varAll.columns.levels[0]
AUnames=varAll.columns.levels[1]
#choose labeling:
#y_true=Labels.mean(axis=0,level='subject')
group=Labels.mean(axis=0,level='subject')
GroupBy='Group'
y_true=subjectsDetails[GroupBy]

#calc ROC using categorial model:
model=svm.SVC(kernel='linear')
#model=LogisticRegression()
roc_auc=DF(index=featureTypes,columns=['auc'])
y_true=y_true.loc[varAll.index]
y_true=y_true=='Male'
y_true=y_true*1

#Anova
D=subjectsDetails.loc[varAll.index]
meanOverFeaturesDF=subjectMean.mean(axis=1,level='ftype')
AllDF=concat([meanOverFeaturesDF,D['Group'],D['Gender'],D['Age'],D['ReligionCode'],D['Education']],axis=1)
AllDFControls=AllDF.loc[AllDF['Group']==0]
AllDFPatients=AllDF.loc[AllDF['Group']==1]
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
AllDF=AllDF.drop('Groups',axis=1)
featureTypes=AllDF.columns.drop('Group')
featureSTR=''
for f in featureTypes:
    featureSTR=featureSTR+' + ' +f
featureSTR=featureSTR[3:]

#cw_lm=ols('Group  ~ Gender + Age + ReligionCode +Education', data=AllDF).fit() #Specify C for Categorical
cw_lm=ols('Group  ~ ' + featureSTR, data=AllDF).fit() #Specify C for Categorical
print(sm.stats.anova_lm(cw_lm, typ=2))



coefs={}
for f in featureTypes:
    #var=varAll.fillna(0)
    print(f)
    var=subjectMean[f].fillna(0)
    model.fit(var,y_true)
    coefs[f]=model.coef_.flatten()
    y_proba=model.decision_function(var)
    y_prediced=model.predict(var)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc.loc[f] = auc(fpr, tpr)
coefsDF=DF.from_dict(coefs)
roc_auc.columns.names=[GroupBy + ' auc']
coefsDF=DF.from_dict(coefs)
coefsDF.index=var.columns
print(roc_auc)

#calc test for two groups
DependentVar='FastChangeRatio'#raw_input('mean over what? enter specific feature type or AU number:  ')
IndependentVars=AUnames

groups=y_true #y_true[0]
try:
    var=varAll[DependentVar]
except KeyError:
    var=varAll.swaplevel(0,1,axis=1)
    var=var[DependentVar]
v1=var.loc[groups==1]
v0=var.loc[groups==0]
if type(v1.values[0]) is str:
    v0=DF([int(s) for s in v0.values.flatten()],index=v0.index)
    v1=DF([int(s) for s in v1.values.flatten()],index=v1.index)
v0=v0.dropna()
v1=v1.dropna()
n0=len(v0)
n1=len(v1)
mean0=v0.mean()
mean1=v1.mean()
sterr0=v0.std()/np.sqrt(len(v0))
sterr1=v1.std()/np.sqrt(len(v1))
t,p=ttest(v0.dropna(),v1.dropna())
ttestDF=DF([mean0,mean1,sterr0,sterr1,n0,n1,t,p],index=['mean0','mean1','sterr0','sterr1','n0','n1',"student's-t",'p-val'],columns=[IndependentVars]).T
#ttestDF['pval']=p
print(ttestDF)

#histogram of PANSS severity:
hist=DF(index=PANSSindex,columns=range(1,8))
PANSSpatients=PANSS_elena[y_ture==1]
for p in PANSSindex:
    PANSSp=PANSS_elena[p]
    hist.loc[p]=numpy.histogram(PANSSp,bins=7)[0]

#k-means
from sklearn.cluster import KMeans
from myUtils import *
from myClasses import *
#load data
DataPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\LearningData' 
dataObject=pickle.load(open(os.path.join(DataPath,'DATAraw.pickle'),'rb'))
rawDF=dataObject.rawDF.copy()

#choose k according to distance of sample from cluster center:
kRange=range(1,30)
DistanceFromCenters={}
for k in kRange:
    print(k)
    kmeans_k=KMeans(n_clusters=k)
    kmeans_k.fit_predict(rawDF)
    DistanceFromCenters[k]=kmeans_k.inertia_

#calc features using chosen k
n_clusters=7
kmeans=KMeans(n_clusters=n_clusters)
rawDF=rawDF.dropna()
#rawDF=rawDF.drop('time',axis=1).T
kmeans.fit_predict(rawDF)
clusteredDF=DF(kmeans.labels_,index=rawDF.index)
subjectsDetails=DF.from_csv('C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\SubjectsDetailsDF2-fill with data from michael.csv',index_col=[0])
Labels=subjectsDetails['Group']
subjectsList=list(set(clusteredDF.index.get_level_values('sCode')))
clusterCountsIndex=[str(c)+'_counts' for c in range(n_clusters)]
clusterLengthIndex=[str(c)+'_length' for c in range(n_clusters)]
clusterNumIndex=[str(c)+'_num' for c in range(n_clusters)]
clustersScores=clusterCountsIndex+clusterLengthIndex+clusterNumIndex+['NumOfClusters','ClusterMeanLength','ClusterChangeRatio']
clustersSummary=DF(index=subjectsList,columns=clustersScores)

for subject in subjectsList:
    subjectClusters=clusteredDF.loc[subject][0]
    value_counts=subjectClusters.value_counts()
    for c in value_counts.index:
        clustersSummary[str(c)+'_counts'].loc[subject]=value_counts.loc[c]
        [Blocks,_]=featuresUtils.countBlocks(DF(subjectClusters==c))
        clustersSummary[str(c)+'_length'].loc[subject]=Blocks['meanBlockLength'][0]
        clustersSummary[str(c)+'_num'].loc[subject]=Blocks['NumOfBlocks'][0]
    
    clustersSummary.loc[subject]['NumOfClusters']=len(value_counts)
    #ClustersTransitionMatrix=featureUtils.getTransMatrix(subjectClusters,n_clusters)   
    TransitionMatrix=featuresUtils.getTransMatrix(DF(subjectClusters),n_clusters)[0]
    N=sum(sum(TransitionMatrix))
    ChangeFrames=N-(sum(np.diagonal(TransitionMatrix)))
    ChangeRatio=ChangeFrames/(N-TransitionMatrix[0,0])
    clustersSummary['ClusterMeanLength'].loc[subject]=clustersSummary[clusterLengthIndex].loc[subject].mean()
    clustersSummary['ClusterChangeRatio'].loc[subject]=ChangeRatio

clustersSummary=clustersSummary.fillna(0.)
clustersSummary['Groups']=Labels[subjectsList]
clustersCenters=DF(kmeans.cluster_centers_,index=range(n_clusters),columns=rawDF.columns)
