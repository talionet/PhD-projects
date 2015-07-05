import numpy as np

from pandas import *
from pandas import DataFrame as DF

import scipy.cluster as cluster
import scipy.stats as sstats

from sklearn import svm
from sklearn import linear_model
from sklearn import cross_validation as cross_val
from sklearn import feature_selection as f_selection
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

import os
import pickle
import matplotlib.pyplot as plt
import globalVars


from myUtils import *
from myClasses import *

class newObject(object):
    pass
##

"""--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Learning Class - TODO- Write details HERE
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------""" 
class LearnObject: 

    def __init__(self,FeatureObject,LabelsObject,LabelsObject2='notDefined'):
        self.FeaturesDF=FeatureObject.FeaturesDF
        self.LabelsObject=LabelsObject
        self.LabelsObject2=LabelsObject2
        self.Details={'LabelDetails':LabelsObject.LabelingDetails,'stratifiedKFold':FeatureObject.details,'FeatureMethod':FeatureObject.method,'PieceLength':FeatureObject.details['PieceLength']}
        self.BestFeatures={}
        self.N=LabelsObject.N
        self.model='notDefined'
        
    
    class BestFeaturesForLabel(): #class of the best features for certain Labeling method (PatientsVsContols, mentalStatus, PANSS, etc.)
        def __init__(self,FeatureTypeList,LabelingList,n_features):
            self.df=DF(np.zeros([len(FeatureTypeList),n_features]),index=MultiIndex.from_tuples(FeatureTypeList),columns=range(n_features))            
            
        def add(self,bestNfeatures): #adds a feature to best features list (length n_features)   
            BestFeaturesList=[j for j in bestNfeatures]
            FeatureTypeList=self.df.index
            for feature in FeatureTypeList:
                if feature in BestFeaturesList:
                    isFeature=1
                    FeatureLoc=BestFeaturesList.index(feature)
                    self.df.loc[feature][FeatureLoc] +=1 
                 
    """def analyzeFeaturesWeight(BestFeaturesDF,weights,ByLevel=0): #after having n features, this analyzes the wheighted mean of the use in each feature type. 
        df=BestFeaturesDF 
        #N=df.sum().sum()
        dfSum=df.sum(level=ByLevel)
        self.Mean=dfSum.sum(axis=1)
            
        weights=self.weights#[1.0/(x+1) for x in df.columns]            
        wSum=dfSum.mul(weights)
        wN=wSum.sum().sum()
        self.WeightedMean=wSum.sum(axis=1)/wN
        return WeightedMean""" 

        #TODO -> add analysis according to facial part (according to excel..)
            #TODO - > add analysis according to learning weights (and not 0.1 : 0.9)
                 
    def run(self,Model='ridge',kernel='linear', cross_validationMethod='KFold',FeatureSelection='PCA',n_features=20,scoringList=['specificity','sensitivity','precision','f1','accuracy','ss_mean'],isSaveCsv=None,isSavePickle=None, isSaveFig=None, isPerm=0,isBetweenSubjects=True,isConcatTwoLabels=False):       
        # -- TODO :
        # --  # Greedy selection on features + Other feature selection types...
        # --  # Make sure featuers are Best only based on train data!!!
        # --  # Keep a list of n_train, n_test from each Label and scoring (accuracy, f1..) in each cross validation iteration
        # --  # Plot results summary (see CARS paper for desired results for Ein Gedi Poster 22-1-2015)
        # --  # remove irelevant data using 'Tracking Success' and consider 'TimeStamps' for feature calculation
        # --  # add f feature analysis by facial part (see excel) 
        # --  # select best model (svm, otherwise ridge regression) 
        # --  # compare svc results with regerssion results (using LOO and different Params for regression  - params for unbalanced data, different kernels, etc.), model evaluation - http://scikit-learn.org/stable/modules/model_evaluation.html) 
        # --  # check how the model weights behave - feature selection analysis
        # --  # calc model error
        # --  # divide data to subparts for training and testing - try within/ between subject, and analyze distribution of features when data is divided
        # --  # LOO - also on bool labels (patients vs controls and mental status bool)
        # --  # add mental status rank scores (0-4)
        # --  # make sure p-val returns the right value in 'scores'
        # --  # run it over random data (permutation test) 
        # --  # continoue here - check regression results-Make sure regression works (not so good).. check what happens in svc for G7 (high train R, negative test R)

        ## init        
        FeatureTypeList=[j for j in tuple(self.FeaturesDF.index)]
        self.FullResults=DF()
        self.Learningdetails={'Model':Model,'Kernel':kernel,'CrossVal':cross_validationMethod,'FeatureSelection':FeatureSelection,'LabelBy':self.Details['LabelDetails'].keys()[0],'FeatureMethod':self.Details['FeatureMethod'],'PieceLength':self.Details['PieceLength']}
        print('\n------------Learning Details------------')
        print(DF.from_dict(self.Learningdetails,orient='index'))
        print('\n----' + cross_validationMethod + ' Cross validation Results:----')
           
        # Set learning params (cross validation method, and model for learning)
        isBoolLabel=self.LabelsObject.isBoolLabel
        isBoolScores=isBoolLabel
        model, isBoolModel, featureSelectionMethod,selectFeaturesFunction= learningUtils.setModel(Model,FeatureSelection,n_features)
        #define global variables over modules (to be used in myUtils)
        globalVars.transformMargins=0#lambda x:x         
        globalVars.isBoolLabel=isBoolLabel
        globalVars.isBoolModel=isBoolModel
        global trainLabels_all, testLabels_all, TrueLabels,isAddDroppedSubjects 
        trainLabels_all, testLabels_all, TrueLabels,isAddDroppedSubjects=labelUtils.initTrainTestLabels_all(self.LabelsObject)
        trainLabels_all2, testLabels_all2, TrueLabels2,isAddDroppedSubjects2=labelUtils.initTrainTestLabels_all(self.LabelsObject2)



        
        
        
        LabelingList=['N1']#trainLabels_all.columns
        self.ResultsDF=DF()
        self.BestFeatures=DF(columns=LabelingList) #dict of BestFeaturesDF according to Labeling methods
        YpredictedOverAllLabels=pandas.Panel(items=range(len(trainLabels_all)),major_axis=LabelingList,minor_axis=TrueLabels.index) #panel: items=cv_ind, major=labels, minor=#TODO 
       
                                              
        ## Create train and test sets according to LabelBy, repeat learning each time on different Labels from LabelingList.
        for label_ind, Labeling in enumerate(LabelingList):
            """if isPerm: #TODO - fix this to work with continous / bool data
                try:
                    trainLabels=self.LabelsObject.permedLabelsDF[Labeling]
                except AttributeError:
                    self.LabelsObject.permLabels()
                    trainLabels=self.LabelsObject.permedLabelsDF[Labeling]"""
            #set subjects list according to labels and features
            X,SubjectsList,droppedSubjects,Xdropped=featuresUtils.initX(self.FeaturesDF,trainLabels_all,Labeling)
            X2,SubjectsList2,droppedSubjects2,Xdropped2=featuresUtils.initX(self.FeaturesDF,trainLabels_all2,Labeling,is2=1)
            
            #init train and test labels
            trainLabels, testLabels, LabelRange = labelUtils.initTrainTestLabels(Labeling,SubjectsList,trainLabels_all, testLabels_all)
            trainLabels2, testLabels2, LabelRange2 = labelUtils.initTrainTestLabels(Labeling,SubjectsList2,trainLabels_all2, testLabels_all2)
            
            #make sure only labeled subjects are used for classification
            X=X.query('subject == '+ str(list(trainLabels.index)) ) 
            X.index.get_level_values(X.index.names[0]) 
            SubjectIndex=list(set(X.index.get_level_values('subject')))

            X2=X2.query('subject == '+ str(list(trainLabels2.index)) )  
            X2.index.get_level_values(X2.index.names[0]) 
            SubjectIndex2=list(set(X2.index.get_level_values('subject')))                       
            #init vars
            if isBetweenSubjects:
                cv_param=len(SubjectIndex)
                self.Learningdetails['CrossValSubjects']='between'
                isWithinSubjects=False
            else:
                isWithinSubjects=True
                X=X.swaplevel(0,1)
                PieceIndex=list(set(X.index.get_level_values('Piece_ind')))
                cv_param=len(PieceIndex)
                self.Learningdetails['CrossValSubjects']='within'
            
            self.Learningdetails['NumOfFeatures']=n_features
            
            print('\n**' + Labeling + '**')
            
            cv, crossValScores= learningUtils.setCrossValidation(cross_validationMethod,cv_param,trainLabels,isWithinSubjects) 
            
            ## Learning - feature selection for different scoring types, with cross validation - 

            BestFeaturesForLabel=self.BestFeaturesForLabel(FeatureTypeList,LabelingList,n_features) #saves dataframe with best features for each label, for later analysis
            cv_ind=0
            #used for transforming from margins returned from svm to continouse labels (e.g . PANSS)
            trainScores=DF()
            test_index=X.index
            testScores=concat([DF(index=test_index),DF(index=['std_train_err'])])
            testScores2=concat([DF(index=testLabels.index),DF(index=['std_train_err'])]) 
            #impt=Imputer(missing_values='NaN', strategy='median', axis=0)

            globalVars.LabelRange=LabelRange

            ModelWeights1=DF(columns=range(len(cv)),index=X.columns)
            Components=pandas.Panel(items=range(len(cv)),major_axis=X.columns,minor_axis=range(n_features)) #todo fix this for 1st and second learning
            ExplainedVar=DF(columns=range(len(cv)))
            ModelWeights2=DF(columns=range(len(cv)))
            for train, test in cv:

                if isBetweenSubjects:
                    #set X and Y
                    train_subjects=trainLabels.iloc[train].index
                    test_subjects=testLabels.iloc[test].index 
                    Xtrain,Xtest, Ytrain, YtrainTrue, Ytest=learningUtils.setXYTrainXYTest(X,Labeling,trainLabels,testLabels,TrueLabels,train_subjects,test_subjects)
                    Xtrain2,Xtest2, Ytrain2, YtrainTrue2, Ytest2=learningUtils.setXYTrainXYTest(X2,Labeling,trainLabels2,testLabels2,TrueLabels2,train_subjects,test_subjects)

                    
                    if isConcatTwoLabels: #used when there is more than one doctor
                        Xtrain=concat([Xtrain,Xtrain2])
                        Xtest=concat([Xtest,Xtest2])
                        Ytrain=concat([Ytrain,Ytrain2])
                        YtrainTrue=concat([YtrainTrue,YtrainTrue2])
                        Ytest=concat([Ytest,Ytest2])
                        Xdropped=concat([Xdropped,Xdropped2])
                        SubjectsList=list(set(SubjectsList).intersection(set(SubjectsList2)))
                        droppedSubjects=list(set(droppedSubjects).union(set(droppedSubjects2)).difference(set(SubjectsList)))#diff from SubjectsList to make sure no subjects are both in train and test.
                    """else:
                        Xtrain=Xtrain1
                        Xtest=Xtest1
                        Xdropped=Xdropped1
                        Ytrain=Ytrain1
                        YtrainTrue=YtrainTrue1
                        Ytest=Ytest1"""

                    #select N best features:
                    Xtrain, Xtest, bestNfeatures, components, explainedVar,decomposeFunc=learningUtils.selectBestNfeatures(Xtrain,Xtest,Ytrain,n_features,selectFeaturesFunction)
                    BestFeaturesForLabel.add(bestNfeatures) #todo - delete this??     

                    #train 1 
                    TrainModel=model
                    TrainModel.fit(Xtrain.sort_index(),Ytrain.T.sort_index())
                    try:
                        Components[cv_ind]=components.T
                        ExplainedVar[cv_ind]=explainedVar
                        isDecompose=True
                        if cv_ind==0:
                            ModelWeights1=DF(columns=range(len(cv)),index=range(len(bestNfeatures)))
                        ModelWeights1[cv_ind]=TrainModel.coef_.flatten()
                    except AttributeError:
                        isDecompose=False
                        ModelWeights1[cv_ind].loc[bestNfeatures]=TrainModel.coef_.flatten()
                    self.isDecompose=isDecompose                    
                    #train 2
                    if isBoolLabel:
                       PiecePrediction_train=DF(TrainModel.predict(Xtrain),index=Xtrain.index,columns=['prediction'])
                       TrainModel2=svm.SVC(kernel='linear', probability=True,class_weight={0:1,1:1})
                    else:
                       PiecePrediction_train=DF(TrainModel.decision_function(Xtrain),index=Xtrain.index,columns=['prediction'])
                       TrainModel2=linear_model.LinearRegression()

                    Xtrain2, Ytrain2, YtrainTrue2=learningUtils.getX2Y2(Xtrain,Ytrain,YtrainTrue,PiecePrediction_train, isBoolLabel)                 
                    TrainModel2.fit(Xtrain2, Ytrain2)
                    if cv_ind==0:
                        ModelWeights2=DF(columns=range(len(cv)),index= Xtrain2.columns)
                    ModelWeights2[cv_ind]=TrainModel2.coef_.flatten()         

                              
                    #test 1
                    if isAddDroppedSubjects: #take test subjects from cv + subjects that were dropped for labeling used for test
                        if isDecompose:
                            dXdropped=DF(decomposeFunc(Xdropped).values,index=Xdropped.index)
                        XtestDropped=dXdropped[bestNfeatures]
                        YtestDropped=Series(XtestDropped.copy().icol(0))
                        #YTrueDropped=Series(Xdropped.copy().icol(0))
                        for subject in droppedSubjects:
                            YtestDropped[subject]=testLabels_all[Labeling].loc[subject]
                            #YTrueAll.loc[subject]=TrueLabels[Labeling].loc[subject]
                        Ytest=concat([Ytest,YtestDropped]).sort_index()
                        Xtest=concat([Xtest,XtestDropped]).sort_index()


                    if isPerm: #TODO- Check this!!
                        Ytest=y_perms.loc[Ytest.index]
                    Xtest=Xtest.fillna(0.)
                    
                    
                elif isWithinSubjects:
                    #train 1
                    train_pieces=PieceIndex[train]
                    test_pieces=PieceIndex[test] #TODO - make sure that if test/train> piece index, it ignores it and repeate the process
                    
                    XtrainAllFeatures=X.query('Piece_ind == '+ str(list(train_pieces)))
                    Ytrain=Series(index=X.index)
                    Ytest=Series(index=X.index)
                    YtrainTrue=Series(index=X.index)
                    
                    for subject in PieceIndex: 
                        for piece in train_pieces:
                            Ytrain.loc[piece].loc[subject]=trainLabels[subject]
                            YtrainTrue.loc[piece].loc[subject]=TrueLabels[Labeling].loc[subject] 
                            Ytest.loc[piece].loc[subject]=testLabels[subject]   
                    Ytrain=Ytrain.dropna()
                    YtrainTrue=YtrainTrue.dropna() 
                    for subject in test_subjects:
                        Ytest.loc[piece].loc[subject]=testLabels[subject]
                #train scores 1       
                if cv_ind==0:
                    trainScores,YtrainPredicted=learningUtils.getTrainScores(Ytrain,Xtrain,YtrainTrue,TrainModel)
                    plt.figure(1)
                    if len(LabelingList)>1:
                        plt.subplot(round(len(LabelingList)/2),2,label_ind+1)
                    if isBoolLabel:
                        testScores=learningUtils.getTestScores(Ytest,Xtest,TrainModel)
                    else:
                        testScores[cv_ind]=learningUtils.getTestScores(Ytest,Xtest,TrainModel)
                        plt.title(Labeling,fontsize=10)
                else:
                    plt.figure(3)
                    new_trainScores,YtrainPredicted=learningUtils.getTrainScores(Ytrain,Xtrain,YtrainTrue,TrainModel)
                    trainScores=concat([trainScores,new_trainScores],axis=1)
                #test 1   
                    testScores[cv_ind]=learningUtils.getTestScores(Ytest,Xtest,TrainModel)
                
                #train2

                if isBoolLabel:
                    PiecePrediction_test=DF(TrainModel.predict(Xtest),index=Xtest.index,columns=['prediction'])
                else:
                    PiecePrediction_test=DF(TrainModel.decision_function(Xtest),index=Xtest.index,columns=['prediction'])
                Xtest2, Ytest2 , YtestTrue2 =learningUtils.getX2Y2(Xtest,Ytest,Ytest,PiecePrediction_test, isBoolLabel)
                
                if cv_ind==0:
                    trainScores2,YtrainPredicted2=learningUtils.getTrainScores(Ytrain2,Xtrain2,YtrainTrue2,TrainModel2)
                    YpredictedOverAllLabels[cv_ind].loc[Labeling]=YtrainPredicted2
                    #plt.figure(1)
                    #if len(LabelingList)>1:
                        #plt.subplot(round(len(LabelingList)/2),2,label_ind+1)
                #test2
                    if isBoolLabel:
                        testScores2=learningUtils.getTestScores(Ytest2,Xtest2,TrainModel2)
                    else:
                        testScores2[cv_ind]=learningUtils.getTestScores(Ytest2,Xtest2,TrainModel2)
                    #plt.title(Labeling,fontsize=10)
                else:
                    new_trainScores2,YtrainPredicted2=learningUtils.getTrainScores(Ytrain2,Xtrain2,YtrainTrue2,TrainModel2)
                    YpredictedOverAllLabels[cv_ind].loc[Labeling]=YtrainPredicted2
                    trainScores2=concat([trainScores2,new_trainScores2],axis=1)
                    testScores2[cv_ind]=learningUtils.getTestScores(Ytest2,Xtest2,TrainModel2)     
                cv_ind+=1

                #crossValScores=crossValScores.append(CVscoresDF,ignore_index=True) #information about entire train test data. 
            fig2=plt.figure(2)
            if len(LabelingList)>1:
                plt.subplot(round(len(LabelingList)/2),2,label_ind+1)
            #if isAddDroppedSubjects:
               # testLabelsSummary=testLabels_all[Labeling].loc[AllSubjects]
           # else:
               # testLabelsSummary=testLabels
            scoresSummary = learningUtils.getScoresSummary(trainScores2,testScores2,TrueLabels[Labeling])
            # reset global vars
            globalVars.fitYscale='notDefined'
            globalVars.beta=DF()

            plt.title(Labeling,fontsize=10)
            plt.xlabel('Ytrue',fontsize=8)
            plt.ylabel('Ypredicted',fontsize=8)
            plt.tick_params(labelsize=6)
            #print(crossValScores.T)    
            scores=scoresSummary.fillna(0.)
            
            #analyze feature weightsL

            WeightedFeatures1=DF([ModelWeights1.mean(axis=1),ModelWeights1.std(axis=1)],index=['mean','std']).T.fillna(0)
            if isDecompose==0:
                WeightedFeatures1FeatureType=WeightedFeatures1.mean(level='FeatureType')
                WeightedFeatures1FsSingal=WeightedFeatures1.mean(level='fs-signal')
                WeightedFeatures1=concat([DF(index=['-------(A) FeatureType-------']),WeightedFeatures1FeatureType,DF(index=['-------(B) faceshift signal-------']),WeightedFeatures1FsSingal])
            
            WeightedFeatures2=DF([ModelWeights2.mean(axis=1),ModelWeights2.std(axis=1)],index=['mean','std']).T.fillna(0)
            BestFeatures=concat([DF(index=['------------- Learning 1 -------------']),WeightedFeatures1,DF(index=['------------- Learning 2 -------------']),WeightedFeatures2])
            self.BestFeatures[Labeling]=BestFeatures['mean']

            #analyze decomposition
            if isDecompose:
                Components_mean = Components.mean(axis=0)
                Components_std = Components.std(axis=0)
                ExplainedVar_mean = DF(ExplainedVar.mean(axis=1)).T#todo- check!
                ExplainedVar_mean.index=['ExplainedVar_mean']
                ExplainedVar_std = DF(ExplainedVar.std(axis=1)).T#todo- check!
                ExplainedVar_std.index=['ExplainedVar_std']
                try:
                    self.LabelComponents[Labeling]=concat([DF(index=['---components mean---']),Components_mean,ExplainedVar_mean,DF(index=['---components std over cross validation---']),Components_std,ExplainedVar_std])
                except AttributeError:
                    self.LabelComponents=dict.fromkeys(LabelingList)
                    self.LabelComponents[Labeling]=concat([DF(index=['---components mean---']),Components_mean,ExplainedVar_mean,DF(index=['---components std over cross validation---']),Components_std,ExplainedVar_std])

                """print(Components_mean)
                print(ExplainedVar_mean)
                print(WeightedFeatures1)"""

                        
            #BestFeaturesForLabel.analyze(ByLevel=0) #TODO change to regression coeff
            LabelFullResults=concat([DF(index=[Labeling]),scores]) 
  
            self.FullResults=concat([self.FullResults,LabelFullResults])            
            self.ResultsDF=concat([self.ResultsDF,DF(scores[0],columns=[Labeling])],axis=1)
#continue here!! to build pseudo inverse matrix from predicted to true - make sure columns + rows are set!

            #self.BestFeatures[Labeling]=BestFeaturesForLabel.WeightedMean

            #plt.savefig('C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\'+Labeling+'png')
        testScores3=pandas.Panel(items=range(len(X2.index))) #for each cv score...
        FullSubjectsList=YpredictedOverAllLabels[0].columns
        YdroppNans=YpredictedOverAllLabels.dropna(axis=0,how='all')
        YdroppNans=YdroppNans.dropna(axis=1,how='all')
        YpredictedOverAllLabels=YdroppNans.dropna(axis=2,how='all')
        notNans_cv_ind=YpredictedOverAllLabels.items
        notNans_trainSubjects=YpredictedOverAllLabels.minor_axis
        notNans_LabelsList=YpredictedOverAllLabels.major_axis
        notNans_TrueLabels=TrueLabels.T[notNans_trainSubjects].loc[notNans_LabelsList]
        cv_ind=0
        for train, test in cv:
            if cv_ind in notNans_cv_ind:
                print(test)
                train=list(set(FullSubjectsList[train]).intersection(set(notNans_trainSubjects)))
                test=list(set(FullSubjectsList[test]).intersection(set(notNans_trainSubjects)))
                if len(train)>0 and len(test)>0: 
                    AllLabelsYTrainPredicted=YpredictedOverAllLabels[cv_ind][train]
                    AllLabelsYTrainPredicted=AllLabelsYTrainPredicted.fillna(0)
                    AllLabelsYTrainTrue=notNans_TrueLabels[train]
                    AllLabelsYTestPredicted=YpredictedOverAllLabels[cv_ind][test]
                    AllLabelsYTestTrue=notNans_TrueLabels[test]

                    pseudoInverse_AllLabelsYTrainTrue=DF(np.linalg.pinv(AllLabelsYTrainTrue),columns=AllLabelsYTrainTrue.index,index=AllLabelsYTrainTrue.columns)
                    global AllLabelsTransformationMatrix
                    AllLabelsTransformationMatrix=DF(AllLabelsYTrainPredicted.dot(pseudoInverse_AllLabelsYTrainTrue),columns=pseudoInverse_AllLabelsYTrainTrue.columns)#change to real code!!
                TrainModel3=lambda y: y.T.dot(AllLabelsTransformationMatrix)
                testscores3[cv_ind]=learningUtils.getTestScores(AllLabelsYTrainTrue,AllLabelsYTrainPredicted,TrainModel3)
            cv_ind+=1

           
        self.ResultsDF=self.ResultsDF.fillna(0.)  
        
        ## Print and save results  
        print('\n')
        print(self.ResultsDF)
        print('\n')
        D=self.Learningdetails 
        savePath=resultsPath+'\\'+D['Model']+'_'+D['CrossVal']+'_LabelBy'+D['LabelBy']+'_Features'+D['FeatureMethod']+ '_FS'+FeatureSelection+'_Kernel'+D['Kernel']+'_'+D['CrossValSubjects']+'Subjects_PieceSize'+D['PieceLength']
        if isPerm:
            savePath=savePath+'_PERMStest'
        saveName=savePath+'\\'+str(n_features)+'_features'        
        self.Learningdetails['saveDir']=savePath
        dir=os.path.dirname(saveName)
        if not os.path.exists(dir):
            os.makedirs(dir)
        if isSavePickle is None:
            isSavePickle=int(raw_input('Save Results to pickle? '))
        if isSaveCsv is None:
            isSaveCsv=int(raw_input('save Results to csv? '))
        if isSaveFig is None:
            isSaveFig=int(raw_input('save Results to figure? '))

       
        if isSavePickle:        
            self.ResultsDF.to_pickle(saveName+'.pickle')
            self.BestFeatures.to_pickle(saveName+'_bestFeatures.pickle')
                
        if isSaveCsv:
            DetailsDF=DF.from_dict(self.Learningdetails,orient='index')
            ResultsCSV=concat([self.ResultsDF,DF(index=['-------Label Details-------']),self.N,DF(index=['-------Learning Details-------']),DetailsDF,DF(index=['-------Selected Features Analysis------']),self.BestFeatures])
            ResultsCSV.to_csv(saveName+'.csv')

        if isSaveCsv or isSavePickle:
            print('successfully saved as:\n' + saveName)
        
        if isSaveFig:
            plt.figure(1)
            plt.savefig(saveName + 'Train.png')
            plt.figure(2)
            plt.savefig(saveName + 'Test.png')
        plt.close()
        plt.close()
"""--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

MAIN FUNCTION  - TODO- Write details HERE
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""    

def main(isLoadData=1,isCutData=0, PieceLength=500, isLoadFeatures=1,isGetFeaturesNaNs=0, isLoadLabels=1, LabelFileName={}, FeatureMethod='Quantization' ,LabelBy='PANSS'):
    # -- TODO:
    # -- # 
    # -- # make sure it works for len(FeatureMethod)>1, now only uses FeatureTypeList[0]
    os.system('cls')
    
    ## Construct / load DATA object
    DataPath=resultsPath+'\\LearningData' 
    if isLoadData:
        print('loading DATA from '+DataPath+ '...')    
        dataObject=pickle.load(open(os.path.join(DataPath,'DATAraw.pickle'),'rb'))#TODO change 'raw' to PieceLength variable and make sure it loads the cutted data
    else:
        AllAUs= ['TimeStamps','TrackingSuccess','au1','au2','au3','au4','au5','au6','au7','au8','au9','au10','au11','au12','au13','au14','au15','au16','au17','au18','au19','au20','au21','au22','au23','au24','au25','au26','au27','au28','au29','au30','au31','au32','au33','au34','au35','au36','au37','au38','au39','au40','au41','au42','au43','au44','au45','au46','au47','au48']
        GoodTrackableAUs=['au17', 'au18', 'au19', 'au1', 'au22', 'au25', 'au26', 'au27', 'au28', 'au29', 'au2', 'au30', 'au31', 'au32', 'au33', 'au34', 'au37', 'au41', 'au43', 'au45', 'au47', 'au48', 'au8']
        PartNames='Interview'     
        isQuantize=True
        isCutData=True
        #print('fs-signal: ' + GoodTrackableAUs)  
        print('Part: '+ PartNames)
        #print('isQuantize=' + str(isQuantize))
        isSetDataParams=int(raw_input('reset data params? '))
        if isSetDataParams:
            GoodTrackableAUs=raw_input('set fs-signal (as list): ')
            PartNames=raw_input('set Part name (as str, capital first letter): ')
            dataObject=DataObject(PartNames,VarNames=GoodTrackableAUs)
        dataObject.getQuantize()
        pickle.dump(dataObject,open(os.path.join(resultsPath,'DATA'),'wb'))
    
    if isCutData:
        isQuantize=1#raw_input('set isQuantize to: ')
        
        print('constructing Data Object...')    
        print('cutting raw data')
        dataObject.rawDF=dataObject.cutData(dataObject.rawDF,PieceLength)
        print('cutting quantized data')
        dataObject.quantizedDF=dataObject.cutData(dataObject.quantizedDF,PieceLength)
        isSaveData=1#int(raw_input('save cutted data? '))

        if isSaveData:
            saveName=os.path.join(resultsPath ,'LearningData','DATA_'+ str(PieceLength))
            #dataObject.rawDF.to_csv(saveName+'rawDF.csv')
            #dataObject.quantizedDF.to_csv(saveName+'quantizedDF.csv')
            pickle.dump(dataObject,open(saveName +'.pickle','wb'))

    ## Calc / Load FEATURES for learning 
    FeaturesPath=resultsPath + '\\LearningFeatures\\' + FeatureMethod + '_Features_'+str(PieceLength)
    Features=FeatureObject(dataObject,FeaturesPath,PieceLength)
    if isLoadFeatures:
        print('loading FEATURES from '+ FeaturesPath + '...\n')  
        Features.FeaturesDF=read_csv(FeaturesPath+'DF.csv', index_col=[0,1], skipinitialspace=True, header=[0,1])
        Features.method=FeatureMethod
    else:
        if not FeatureMethod:
            FeatureMethod = raw_input("Enter Feature Type ('Quantization', Moments') as list: ")
        print("Calculating subjects' " + FeatureMethod + " features ...")            
        Features.getFeatures(FeatureMethod)
    if isGetFeaturesNaNs:
        Features.FeaturesDF=featuresUtils.getMissingFeatures(Features)
        Features.FeaturesDF.to_csv(Features.FeaturesPath +'DF.csv')
    
    # Set /Load LABELS for Learning
    LabelsPath=resultsPath + '\\LearningLabels\\' + LabelBy + '_Labels' #for loading / saving
    LabelsPath2=LabelsPath+'2'
    if isLoadLabels:
        print('loading LABELS from '+LabelsPath+ '...\n')  
        Labels=pickle.load(open(LabelsPath+".pickle",'rb'))
        Labels2=Labels#pickle.load(open(LabelsPath2+".pickle",'rb')) #todo - change this when there is second labeled data (from michael)
    else:
        Labels=LabelObject(SubjectsDetailsDF,LabelsPath)
        Labels.getLabels(LabelBy)
        SubjectsDetailsDF2=DF.from_csv('C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\SubjectsDetailsDF2-fill with data from michael.csv')
        Labels2=LabelObject(SubjectsDetailsDF2,LabelsPath2)
        Labels2.getLabels(LabelBy)
        #Labels.permLabels() #TODO - move this to "not isLoad" or somewhere else. 
       

    # Get cross validation learning results : 
    # loop  over feature number

    FeatureRange= [10]#range(1,6)#range(1,50,5) #[6]
    #init 
    isBoolLabel=Labels.isBoolLabel
    FeatureComparession={}
    SelectedFeaturesComparession={}
    newDF=lambda:DF(columns=FeatureRange,index=Labels.names)
    if isBoolLabel:
        All_specificity=newDF()
        All_sensitivity=newDF()
        All_precision=newDF()
        All_accuracy=newDF()
        All_f1=newDF()
        All_ss_mean=newDF()
    else:
        All_trainR=newDF()
        All_trainPval=newDF()
        All_trainErr=newDF()
        All_testR=newDF()
        All_testPval=newDF()
        All_testErr=newDF()
        All_testErrStd=newDF()
        All_LabelRange=newDF()

    
    ModelList=['ridge']#['regression','ridge','lasso']
    FeatureSelectionList=['PCA']#,'KernelPCA','SparsePCA','ICA']
    for m in ModelList:
        print('************************************ Model = ' +m+'************************************')
        for fs in FeatureSelectionList:
            print('***************************** FeatureSelection = ' +fs+'******************************')
            for f in FeatureRange:
                print('Num Of Features = ' + str(f))
                s=LearnObject(Features,Labels,Labels2)
                s.run(Model=m,n_features=f,isSavePickle=0,isSaveCsv=1,isSaveFig=1,isPerm=0,isBetweenSubjects=True,FeatureSelection=fs)
                LabelNameList=s.ResultsDF.columns #TODO - CHANGE THIS!
                for label in LabelNameList:
                    print(label)
                    if f==FeatureRange[0]:
                        FeatureComparession[label]=DF(columns=FeatureRange,index=s.ResultsDF.index)
                        SelectedFeaturesComparession[label]=DF(columns=FeatureRange,index=s.BestFeatures.index)
                    FeatureComparession[label][f]=s.ResultsDF[label]     
                    SelectedFeaturesComparession[label][f]=s.BestFeatures[label]
                    r=s.ResultsDF[label]

                    if isBoolLabel: 
                        All_specificity[f].loc[label]=r['specificity'] 
                        All_sensitivity[f].loc[label]=r['sensitivity'] 
                        All_precision[f].loc[label]=r['precision'] 
                        All_accuracy[f].loc[label]=r['accuracy'] 
                        All_f1[f].loc[label]=r['f1']
                        All_ss_mean[f].loc[label]=r['ss_mean']

                    else:
                        All_trainR[f].loc[label]=r['trainR^2']
                        All_trainPval[f].loc[label]=r['trainPval']
                        All_trainErr[f].loc[label]=r['trainError']
                        All_testR[f].loc[label]=r['testR^2'] 
                        All_testPval[f].loc[label]=r['testPval'] 
                        All_testErr[f].loc[label]=r['testError']
                        All_testErrStd[f].loc[label]=r['testErrorStd']
                        All_LabelRange[f].loc[label]=r['LabelRange']

    
        for label in LabelNameList:
            saveName=s.Learningdetails['saveDir']+'\\'+label+'_ResultsSummary.csv'
            if os.path.exists(saveName):
                isSave=raw_input('the file '+saveName+ ' already exist, \noverwrite existing file? ')
            else:
                isSave=1
            if isSave:
                resultsSum=concat([DF(index=['----------- Learning results -----------']),FeatureComparession[label],DF(index=['-------Selected Features Analysis-------']),SelectedFeaturesComparession[label],DF(index=['----------- Learning details -----------']),DF.from_dict(s.Learningdetails,orient='index')])
                if s.isDecompose:
                    resultsSum=concat([resultsSum,s.LabelComponents[label]])
                resultsSum.to_csv(saveName)
    
        if isBoolLabel:               
            ResultsSummary=concat([DF(index=['------specificity vs. Number Of Features-------']),All_specificity,DF(index=['------sensitivity vs. Number Of Features-------']),All_sensitivity,DF(index=['------precision vs. Number Of Features-------']),All_precision,DF(index=['------accuracy vs. Number Of Features-------']),All_accuracy,DF(index=['------f1 vs. Number Of Features-------']),All_f1,DF(index=['------sensitivity-specificity mean vs. Number Of Features-------']),All_ss_mean])
            ResultsSummary.to_csv(s.Learningdetails['saveDir']+'\\ResultsSummary_bool.csv') 
        else:
            ResultsSummary=concat([DF(index=['------train R^2 vs. Number Of Features-------']),All_trainR.dropna(),DF(index=['------train Pval vs. Number Of Features-------']),All_trainPval.dropna(),DF(index=['------train Error vs. Number Of Features-------']),All_trainErr.dropna(),DF(index=['------test R^2 vs. Number Of Features-------']),All_testR.dropna(),DF(index=['------testPval vs. Number Of Features-------']),All_testPval.dropna(),DF(index=['------test test Error vs. Number Of Features-------']),All_testErr.dropna(),DF(index=['------test Error STD vs. Number Of Features-------']),All_testErrStd.dropna(),DF(index=['------Label Range vs. Number Of Features-------']),All_LabelRange.dropna()])
            ResultsSummary.to_csv(s.Learningdetails['saveDir']+'\\ResultsSummary_regression.csv')

            # permutation test:
    """ #init 
    perms=1000
    permsavestep=10
    PermRange=range(perms)
    PermSaveRange=range(permsavestep,perms,permsavestep)
    FeatureComparession={}
    SelectedFeaturesComparession={}
    def initPerms():
        Allf1=DF(columns=PermRange,index=Labels.names)
        Allmargin_R=DF(columns=PermRange,index=Labels.names)
        Allmargin_Pval=DF(columns=PermRange,index=Labels.names)
        Allmargin_stdErr=DF(columns=PermRange,index=Labels.names)
        return Allf1, Allmargin_Pval,Allmargin_R, Allmargin_stdErr

    n_features=10
    sp=0
    SavePerm=PermSaveRange[sp]
    Allf1, Allmargin_Pval,Allmargin_R, Allmargin_stdErr = initPerms()
    for p in PermRange:
        print ('-----------Permutation #'+str(p+1)+'-------------')
        Labels.permLabels(isSavePerms=0)
        s=LearnObject(Features,Labels)
        s.run(n_features=n_features,isSavePickle=0,isSaveCsv=1,isPerm=1)
        for label in LabelsNameList:
            print(label)
            if isBoolLabel: 
                Allf1[p].loc[label]=s.ResultsDF[label]['f1'] 
            else:
                Allmargin_R[p].loc[label]=s.ResultsDF[label]['margins_R'] 
                Allmargin_Pval[p].loc[label]=s.ResultsDF[label]['margins_Pval'] 
                Allmargin_stdErr[p].loc[label]=s.ResultsDF[label]['regression_stdError']

        if p>=SavePerm:          
            ResultsSummary=concat([DF(index=['------R^2 vs. Permutations-------']),Allmargin_R,DF(index=['------Pval vs. Permutations-------']),Allmargin_Pval,DF(index=['------std Error vs. Permutations-------']),Allmargin_stdErr]) 
            ResultsSummary.dropna(axis=1, how='all')
            try:
                ResultsSummary.to_csv(s.Learningdetails['saveDir']+'\\permutation_ResultsSummary_margins'+str(SavePerm)+'.csv')
            except IOError:
        
                isSaveResults=raw_intput('File is open, close it and press 1 to continue or 0 to close without saving ')
                if isSaveResults:
                    ResultsSummary.to_csv(s.Learningdetails['saveDir']+'\\permutation_ResultsSummary_margins'+str(SavePerm)+'.csv')
            sp+=1
            SavePerm=PermSaveRange[sp]
            Allf1, Allmargin_Pval,Allmargin_R, Allmargin_stdErr = initPerms() """
    

    #
   
"""--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""


RawDataPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\AllPartsData'
resultsPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results'
isImport=0
#isImport=int(raw_input('import data? '))

if __name__ == "__main__":
    if isImport:
        AllPartsData=pickle.load(open(RawDataPath+".pickle",'rb'))
    
    SubjectsDetailsDF=pickle.load(open(RawDataPath+"Details.pickle",'rb'))
    #SubjectsDetailsDF2=pickle.load(open(RawDataPath+"Details2.pickle",'rb'))
    permfileDir='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\svc_LOO_LabelByPANSS_FeaturesQuantization_FSdPrime_Kernellinear_PERMStest'
#for p in range(250,950,150):
    #main()
    # todo -
            #BoolPANSS: *svc100, *svc500, *svc1000
            #PatienstVsContols: 
PieceLengthRange=[500]#,1000]
LabelByList=['PANSS']#'PatientsVsControls','boolMentalStatus']
for label in LabelByList:
    print('************************************ LabelBy = ' +label+ '************************************')
    for l in PieceLengthRange:
        print('************************************ PieceLength = ' +str(l)+ '************************************')
        main(PieceLength=l, LabelBy=label)
