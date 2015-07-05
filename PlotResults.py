from pandas import DataFrame as DF
from matplotlib import pyplot as plt
import numpy as np
import os


def barPlotWithStars(var1, pvals):
    # necessary variables
    N=len(var1)
    ind=np.arange(N)
    plt.figure()
    var1.plot(kind='bar')
    plt.ylabel('Pearson R')
    plt.title('Regression Result for PANSS')
    for i, p in enumerate(pvals):
        if p<0.01:
            sigStar='**'
            p_ind=ind[i]-0.14
        elif p<0.05:
            sigStar='*'
            p_ind=ind[i]-0.07
        else:
            sigStar=''
            p_ind=0
        plt.annotate(sigStar, (p_ind,var1[i]))
    return plt


## ---- DEINE AND LOAD VARS  ---- : 
#resultsPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results'
resultsPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\'
savePath='C:\\Users\\taliat01\\Desktop\\TALIA\\Docs\\Papers\\MindCarePaper\\MindCareFigs\\'

# --- set Results Dir --- :
#fileName='ridge_LOO_LabelByPANSS_FSelectionTopNComponents_DecompostionPCA_byFeatureTypePieceSize500_allFeatureTypes\\10_features.csv'
fileName='Figs\\smileAnalysis.csv'
#fileName='svc_LOO_LabelByPatientsVsControls_FSelectionf_regression_DecompostionnoDecompositionPieceSize500_allFeatureTypes\\20_features.csv'
saveNameEnding='smileAnalysis'
saveName=savePath+saveNameEnding+'.jpeg'
results=DF.from_csv(resultsPath+fileName)
results=results.dropna(axis=1,how='all')
results=results.dropna(how='all')
# --- set results Range ---:
#resultsRange='testR^2'
columnRange=['Controls','Patients']#['N1']#results.columns
resultsRange=['FastChangeRatio','ChangeRatio','ExpressionRatio','ExpressionLength','ExpressionLevel']
resultsRangeNames=['Ratio','Level','Length','Change','Fast Change']
#resultsRange=['m1','m2']
var1=DF(index=resultsRange, columns=columnRange)
colResults=results[columnRange]
for r in resultsRange:
    try: 
        var1.loc[r]=colResults.loc[r]
        var1=var1.dropna(axis=1,how='all')
        #pvals=results.loc['roc_auc'].dropna()
    except KeyError:
        var1.loc[r]=0


##  ----- PLOTTING OPTIONS  ----- : 
## scale var to 100:
normalize=lambda x: (x/x.abs().sum())
var1Normalized=var1.apply(normalize)
var1=var1Normalized

# -- Plot Bar With Stars --  : 
plt=barPlotWithStars(var1,pvals)
plt.savefig(saveName)

# -- plot Bar with std for two groups --:
dispOnlySignificant=True
if dispOnlySignificant:
    pvals=results['pval']
    results=results[pvals<0.05]

mean0=results[columnRange[0]]
mean1=results[columnRange[1]]
ste0=results['stderr0']
ste1=results['stderr1']
N=len(mean0)
resultsRange=mean0.index
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects0 = ax.bar(ind, mean0, width, color='darkblue', yerr=ste0,error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
rects1 = ax.bar(ind+width, mean1, width, color='lightblue', yerr=ste1,error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
# add some text for labels, title and axes ticks
ax.set_ylabel('Mean group value',fontsize=14)
ax.set_title('Smile Activity Group Charateristics',fontsize=16)
ax.set_xticks(ind+width)
ax.set_xticklabels( resultsRangeNames,fontsize=14 )
ax.legend((rects0[0], rects1[0]), ('Controls', 'Patients'),loc="upper right")
#plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(saveName)


# -- stacked bar for feature weights -- :
plt.figure()
var1.index=resultsRange#['Segment Prediction Mean','Segment Prediction Std']
var1.T.plot(kind='bar',stacked=True,colormap='BuGn')

plt.legend(bbox_to_anchor=(0.5,1))

plt.title('Regression Weights - Learning II',fontsize=20)
plt.ylabel('Regression Weights')
plt.tight_layout()
plt.savefig(saveName)

# -- horizontal bar plot for svm patients vs. control -- 
plt.figure()
var1.plot(kind='barh',color='darkblue',legend=False)
plt.xlabel('Mean regression weights',fontsize=14)
plt.ylabel('Feature Type',fontsize=14)
plt.title(saveNameEnding)
plt.tight_layout()
plt.savefig(savePath+saveName+'.jpeg')

# -- plot multiple ROC curves -- 
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
saveName='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\Figs\\noLOOBestResults'
resultsPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\svc_NONE_LabelByPatientsVsControls_FSelectionf_regression_DecompostionFeatureType_PCA_byFeatureTypePieceSize500_'
n_features=15
ROCLabels=['ChangeRatio','ExpressionRatio','ExpressionLength','ExpressionLevel','FastChangeRatio']
#plt1=fig.add_subplot(2,1,1)
#plt2=fig.add_subplot(2,1,2)
plt.figure()
aucAll={}
for label in ROCLabels:
    rocDF=DF.from_csv(resultsPath+ label +'\\'+ str(n_features)+'_featuresDF.csv')
    tpr=rocDF['tpr']
    fpr=rocDF['fpr']
    aucAll[label]=auc(fpr,tpr)
    plt.plot(fpr, tpr, linewidth=2, label=label)
plt.plot([0, 1], [0, 1], 'k--')
#plt.setp(color='terrain')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('No-LOO 15 features- fregression Decompostion - PCA_byFeatureType')
plt.savefig(saveName+'ROC.jpeg')

plt.close()
plt.figure()
aucAllDF=DF.from_dict(aucAll,orient='index')
aucAllDF.plot(kind='barh',legend=False,color='c') #todo-fix this!
annotation=aucAll.values.flatten()
annotation_loc=(range(len(annotation)),annotation)
plt.tight_layout()
plt.annotate(annotation, annotation_loc)
#add text AUC over bars
plt.title('NoLOO_FSelectionTopNComponents_Decompostionfs-signal_PCA_byFeatureType')

plt.savefig(saveName) #TODO - fix !!!

plt.close()
#plt.setp(colormap='Blues')



