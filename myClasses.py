import numpy as np
from pandas import *
from pandas import DataFrame as DF

import scipy.cluster as cluster
import scipy.stats as sstats

from sklearn import svm
from sklearn import linear_model
from sklearn import cross_validation as cross_val
from sklearn import feature_selection
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import os
import pickle
import matplotlib.pyplot as plt

from  myUtils import *

class newObject(object):
    pass


"""--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

DATA Class - TODO- Write details HERE
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""        
class DataObject:
    def __init__(self,rawDF,PartNames,VarNames):
        print('Creating data object for '+ str(PartNames)+ '\nVariables : '+ str(VarNames))
        self.rawDF=rawDF#AllPartsData.loc[PartNames,VarNames]
        self.details={'Part':PartNames,'Columns':VarNames}
        self.subjectsList= list(self.rawDF.index.get_level_values('sCode').unique())
        self.ispieceData=0
        self.MethodDetails={}
    
#    def calcSmoothThresh(self,T=0.2):
#        self.smoothed=DF(index=self.rawDF.index)
#        self.MethodDetails['SmoothThresholding']={'T':T}
#        
#        def smooth(x,T=T):   
#            sX= [max(0,abs(xi)-T) for xi in x.values]
#            return sX
#        for columns in self.rawDF:
#            self.Smoothed[columns]=self.rawDF.apply(smooth)
#            
#            plt.plot(range(len(x)),[x,sX])
#            plt.show()
#            print('delete these')
            


"""--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

FEATURE Class - TODO- Write details HERE
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""        
      
class FeatureObject():
     # -- TODO:
     # -- # Add more features
         
    def __init__(self,dataObject,FeaturesPath,PieceLength):
        self.rawDF=dataObject.rawDF
        self.details=dataObject.details
        self.details['PieceLength']=str(PieceLength)
        try:
            self.SubjectsList=list(set(dataObject.rawDF.index.get_level_values('sCode')))
        except KeyError:
            self.SubjectsList=list(set(dataObject.rawDF.index.get_level_values('subject')))

        self.varNames=list(dataObject.rawDF.columns)
        self.data=dataObject
        self.isSave=1#int(raw_input('save features? '))
        self.FeaturesPath=FeaturesPath
        
    def getFeatureDF(self,sData, func, featureName={}):
        FeatureData=np.array(sData.apply(func))
        FeatureIndex=MultiIndex.from_product([featureName,self.varNames],names=['FeatureType','fs-signal'])
        FeatureDF=DF(FeatureData,index=FeatureIndex)
        return FeatureDF
        
    #calc specific features according to feature type:    
    def getFeatures(self,FeatureMethod,cross_validationMethod):
        if not FeatureMethod:
           FeatureMethod=raw_input('Choose feature method (Moments,Quantization) : ')        
        
        print('\nCalculating Features for each subject...')
        #init
        FeaturesArray=np.array([])
        self.FeaturesDF=DF()        
        self.method=FeatureMethod
        #loob over subjects
        PieceSize=int(raw_input('Piece Size= (CHANGE THIS TO MANUAL IN myClasses 125) '))
        for subject in self.SubjectsList:
            print(subject)
            subjectData=self.data.rawDF.loc[subject] #raw subject data 
            fs_signal=subjectData.columns #continue here- make sure the features and labels are sink in different piece_ind.
            #if isCuttedData
                #piecesIndex=list(set(subjectData.index.get_level_values('Piece_ind')))
                #piecesNames=piecesIndex
            #else:
            numOfPieces=range(np.round(len(subjectData)/PieceSize))[:-1]
            piecesIndexDict={}
            piecesName={}
            for i in numOfPieces:
                piecesName[i]=PieceSize*i
                piecesIndexDict[piecesName[i]]=range(PieceSize*i,PieceSize*(i+1))
                subjectDataCutted=subjectData
            piecesIndex=list(piecesIndexDict.itervalues())
            piecesName=list(piecesName.itervalues())
            #
            subjectPiecesIndex=MultiIndex.from_product([self.SubjectsList,piecesName],names=['subject','Piece_ind'])
            #loob over pieces
            for piece in piecesIndexDict:
                pieceRange=piecesIndexDict[piece] #change this to PieceName, if data is already cutted
                sData=subjectData.loc[pieceRange]
                       
            #calc features
                if  FeatureMethod=='Moments' :
                    FeatureTypeList=  ['M1','M2','Skew','Kurtosis']  
                    M1=self.getFeatureDF(sData,np.mean,'m1')
                    M2=self.getFeatureDF(sData,np.std,'m2')
                    Skew=self.getFeatureDF(sData,sstats.skew,'Skew')
                    Kurtosis=self.getFeatureDF(sData,sstats.kurtosis,'Kurtosis')
                    subjectFeatures=concat([M1,M2,Kurtosis,Skew])
                    
                    if self.FeaturesDF.empty:
                        self.FeaturesDF=DF(columns=subjectPiecesIndex, index=subjectFeatures.index)              
                    self.FeaturesDF[subject,piece]=subjectFeatures  
                
                
                if  FeatureMethod=='Quantization':
                    FeatureTypeList=['ExpressionRatio','ExpressionLevel','ExpressionLength','ChangeRatio','FastChangeRatio']
                    #                if 'quantizedDF' in self.data: %TODO- fix this
    #                    self.data.calcQuantize()
                    k=self.data.MethodDetails['Quantize']['NumOfQuants']                
                    qData=self.data.quantizedDF #TODO, make sure it is also divided to pieces
                    sData=qData.loc[subject].loc[pieceRange] # subject data
                    cols=qData.columns
                    
                
                    #calc features using quantized vector:
                    ExpressionRatio,ExpressionLevel,ExpressionLength,ChangeRatio,FastChangeRatio=featuresUtils.calcQuantizationFeatures(sData,k)
                    FeaturesDict={'ExpressionRatio':ExpressionRatio,'ExpressionLevel':ExpressionLevel,'ExpressionLength':ExpressionLength,'ChangeRatio':ChangeRatio,'FastChangeRatio':FastChangeRatio}
                    multInd=MultiIndex.from_product([FeatureTypeList,cols],names=['FeatureType','fs-signal'])
                    subjectFeatures=DF(concat([ExpressionRatio,ExpressionLevel,ExpressionLength,ChangeRatio,FastChangeRatio]).values,index=multInd)
                    if self.FeaturesDF.empty:
                        self.FeaturesDF=DF(columns=sgetubjectPiecesIndex, index=subjectFeatures.index)              
                    self.FeaturesDF[subject,piece]=subjectFeatures 

                if FeatureMethod=='kMeansClustering':
                    n_clusters=7
                    print('clustering data... num of clusters = '+str(n_clusters))
                    rawDataAllSubjects=self.data.cuttedDF
                    rawDataAllSubjectsButOne=rawDataAllSubjects.drop(subject)
                    clusteredDataAllSubjectsButOne,kmeans,clusterCenters=self.data.getClusters(rawDataAllSubjects)
                    subjectRawData=rawDataAllSubjects.loc[subject].loc[pieceRange]
                    subjectClusteredData=kmeans.fit(subjectRawData) #fit k means from train data to test data.4
                    sData=index=DF(subjectClusteredData,rawData.index)
                    subjectFeatures=featuresUtils.calckMeansClusterFeatures(sData,n_clusters)
                    if self.FeaturesDF.empty:
                        self.FeaturesDF=DF(columns=subjectPiecesIndex, index=subjectFeatures.index)              
                    self.FeaturesDF[subject,piece]=subjectFeatures 
        # PreProcessing over all features (normalizeation and dropna())
        GetNormRows=lambda x: (x-x.mean())/x.std()
        self.FeaturesDF=self.FeaturesDF.apply(GetNormRows,axis=1) #normalize each feature over all subjects
        self.FeaturesDF.fillna(0,inplace=True)#make sure there are no inappropriate Nans  #TODO - this should be removed, and get the same FeaturesDF as in 'getMissingFeatures'
        if  FeatureMethod=='Quantization':
            self.FeaturesDF=featuresUtils.getMissingFeatures(self) #calc the real NaNs

        # multIndex=MultiIndex.from_product([FeatureTypeList,fs_signal],names=['FeatureType','fs-signal'])
        # FeaturesArray=StandardScaler().fit_transform(FeaturesArray.T) #normalize each feature over all subjects.
        # FeaturesDF=DF(FeaturesArray.T,index=multIndex,columns=self.SubjectsList)
        #self.FeaturesDF.index.set_names(names=['FeatureType','fs-signal']) #TODO - make sure this works!
      
        if self.isSave:          
            print('\nsaving to pickle...' )
            pickle.dump(self,open(self.FeaturesPath +'.pickle','wb'))
            print('saving to csv...' )
            self.FeaturesDF.to_csv(self.FeaturesPath +'DF.csv')
            print('All Features Data successfully saved to ' +self.FeaturesPath)
            

    
"""--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Label Class - TODO- Write details HERE
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""     
class LabelObject:
    
    def __init__(self,DetailsDF,LabelsPath):
        self.SubjectsDetails=DetailsDF
        self.SubjectsList = list( DetailsDF.index.unique())
        self.boolLabelsDF=DF(index=self.SubjectsList)
        self.contLabelsDF=DF(index=self.SubjectsList)
        self.LabelingDetails={}
        self.LabelingMethod=None
        self.isSave=int(raw_input('save Labels? '))
        self.LabelsPath=LabelsPath

        
    def getLabels(self, LabelingMethod=None, LabelingParams=0,isPerm=0):
       
        if not LabelingMethod:
            LabelingMethod=raw_input('Chooose Labeling Method : 1 - PatientsVsControls ; 2 - boolMentalStatus ; 3 - PANSS :  ')
        print('Labeling subjects by ' + LabelingMethod + '...')
       
        ## Label subjects by Patients VS Controls
        if LabelingMethod == 'PatientsVsControls':
            self.names=['PatientsVsControls'] 
            self.boolLabelsDF=DF.from_dict({self.names[0]: self.SubjectsDetails['Group']})
            self.LabelingDetails['PatientsVsControls']=["According to 'Label' variable in DetailsDF"]
            self.N=DF(self.boolLabelsDF.apply(Series.value_counts))
            self.isBoolLabel=1
#            self.LabelingList=['PatientsVsControls']
         
        ## Label subjects by boolean (0-1) mental status scores
        elif LabelingMethod == 'boolMentalStatus':
            self.names=['posture_normal','posture_comfortable','answer_to_the_point','affect_congruity']
            boolMSData=self.SubjectsDetails.loc[:,self.names]
            self.boolLabelsDF=boolMSData.dropna()
            self.LabelingDetails['boolMentalStatus']=['According to boolean variables in Mental Status']
            self.N=DF(self.boolLabelsDF.apply(Series.value_counts))
            self.isBoolLabel=1

        ## Label subjects by PANSS score
        elif LabelingMethod == 'PANSS':
            self.N=DF()
            self.isBoolLabel=0
            # set Labelin parameters
            if LabelingParams == 0 :
                GP=newObject()                
                GP.lowRange=[1,1]
                GP.highThresh=4
                GP.selectFromLabel0='SubGroup' # ['All']
                GP.selectedPANSS='All'
                GP.PANSSRange=DF()
                GP.PANSSRemovalInfo={}
                LabelingParams=GP
                                
             #take relevant PANSS data from detailsDF            
            if LabelingParams.selectedPANSS == 'All':
                PANSSData=self.SubjectsDetails.loc[:,'P1':'G16']
                PANSSList=list(PANSSData.columns) 
                PANSSRemovalInfo=dict.fromkeys(PANSSList)
            else:   
                PANSSList=LabelingParams.selectedPANSS
                PANSSData=self.SubjectsDetails.loc[:,PANSSList]
                AllPANSSList=list(SubjectsDetails.loc[:,'P1':'G16'].columns)
                PANSSRemovalInfo=dict.fromkeys(AllPANSSList)
                for p in AllPANSSList: 
                    if p not in PANSSList:
                        PANSSRemovalInfo[p]='By User'
            self.names=PANSSList

            #divide into Labels according to selectedPANSS
            for p in PANSSList:
                pData=PANSSData.loc[:,p]
                if max(pData)<LabelingParams.highThresh:
#                    NewPANSSList.remove(p)
                    PANSSRemovalInfo[p]='Symptom range too small for Labeling (max score ='+str(max(pData))+')'
                else: #divide into Labels
                    pMin0=LabelingParams.lowRange[0]
                    pMax0=LabelingParams.lowRange[1]
                    pMin1=max(LabelingParams.highThresh,max(pData)-1)
                    pMax1=max(pData)
                    isLabel0=pData.between(pMin0,pMax0)
                    isLabel1=pData.between(pMin1,pMax1)
                    N0=sum(isLabel0)
                    N1=sum(isLabel1)
                    if N1<2:
#                        NewPANSSList.remove(p)
                        PANSSRemovalInfo[p]='Not enough subjects in Label1 (N1='+str(N1)+' range1=['+str(pMin1)+':'+str(pMin1)+']'
                    else:
                        self.boolLabelsDF[p]=-1                        
                        self.boolLabelsDF[p][isLabel0]=0
                        self.boolLabelsDF[p][isLabel1]=1
                        self.N[p]=[N0,N1,pMin0,pMax0,pMin1,pMax1]
                        del PANSSRemovalInfo[p]
            PANSScols=self.boolLabelsDF.columns
            self.contLabelsDF=DF(PANSSData.loc[:,PANSScols],columns=PANSScols)
            self.N.index=['N0','N1','min0','max0','min1','max1']                       
            LabelingParams.PANSSRemovalInfo=PANSSRemovalInfo
            self.LabelingDetails['PANSS']=LabelingParams.__dict__       
        
             
        if self.isSave:          
            print('\nsaving to pickle...' )
            pickle.dump(self,open(self.LabelsPath +'.pickle','wb'))
            print('saving to csv...' )
            self.boolLabelsDF.to_csv(self.LabelsPath +'DF.csv')
            print('All Labels Data successfully saved to ' +self.LabelsPath)

    def permLabels(self, isSavePerms=1):
        if isBoolLabel==1:
            L=self.boolLabelsDF
        elif isBoolLabel==0:
            L=self.contLabelsDF

        self.permedBoolLabelsDF=DF(index=L.index, columns=L.columns) 
        for col in L.columns:
            self.permedBoolLabelsDF[col]=np.random.permutation(L[col])#TODO - test what happens if the rand perm is on each label seperatly
        try:
           L=self.contLabelsDF
           self.permedContLabelsDF=DF(index=L.index, columns=L.columns)
           for col in L.columns:
            self.permedContLabelsDF[col]=np.random.permutation(L[col])
        except AttributeError:
            pass 

        if 'isSavePerms' not in locals():
            isSavePerms=int(raw_input('save permed Labels? ')      )
        if isSavePerms:
            print('\nsaving to pickle...' )
            pickle.dump(self,open(self.LabelsPath +'.pickle','wb'))
            print('saving to csv...' )
            self.boolLabelsDF.to_csv(self.LabelsPath +'DF.csv')
            print('All Labels Data successfully saved to ' +self.LabelsPath)
                          


