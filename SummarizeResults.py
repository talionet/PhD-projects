from pandas import *
from pandas import DataFrame as DF

def summarizeAllResultsToCsv():
    resultsDir='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\'
    D={}
    # continous list:
    LabelByList=['PatientsVsControls']#'PANSS']
    ModelList=['svc']#'ridge','lasso']
    CrossValList=['LOO','NONE']
    DecompositionList=['noDecomposition']#'PCA_byFeatureType','PCA_byfs-signal_']#,'PCA','None',] #[,'KernelPCA','SparsePCA','ICA']
    FeatureSelectionList=['TopNComponents','f_regression','FirstComponentsAndExplainedVar','FirstComponentsAndFregression','TopExplainedVarianceComponents'] 
    SubFeaturesList=['allFeatureTypes','ChangeRatio','ExpressionRatio','ExpressionLength','ExpressionLevel','FastChangeRatio']
    SegmentSizeList=['500']#'100','500','1000']
    FeatureNumList=['5','8','10','12','15']
    FullResults=DF()
    missingFiles=DF()
    FoundFilesCount=0
    MissingFilesCount=0
    for D['LabelBy'] in LabelByList:
            for D['Model'] in ModelList:
                for D['CrossVal'] in CrossValList:
                    for D['Decomposition'] in DecompositionList:
                        for D['feature_selection'] in FeatureSelectionList:
                            for D['segment_length'] in SegmentSizeList:
                                for D['sub_Feature'] in SubFeaturesList:
                                    for D['featureNum'] in FeatureNumList:
                                        print('\nfound ' + str(FoundFilesCount) + ' out of ' + str(MissingFilesCount+FoundFilesCount)+' files' )
                                        resultsPath=resultsDir + D['Model']+'_'+D['CrossVal']+'_LabelBy'+D['LabelBy']+ '_FSelection'+D['feature_selection']+'_Decompostion'+D['Decomposition']+'PieceSize'+D['segment_length']+'_'+D['sub_Feature']
                                        #dirName=model+mid_dir_name[0]+LabelBy+mid_dir_name[1]+pieceSize
                                        fileName=D['featureNum']+'_features.csv'
                                        filePath=resultsPath+'\\'+fileName
                                        WeightsScores=['au1','au2', 'au8','au17','au18','au19','au22','au25','au26','au27','au28','au29','au30','au31','au32','au33','au34','au37','au41','au43','au45','au47','au48','ChangeRatio','ExpressionRatio','ExpressionLength','ExpressionLevel','FastChangeRatio']
                                        if D['Model']=='svc':
                                            LearningScores=['sensitivity','specificity','roc_auc']
                                       
                                        else: 
                                            LearningScores=['trainR^2','trainPval','testR^2','testPval'] 
                                        try:
                                            rawResults=read_csv(filePath, skipinitialspace=True).dropna(axis=1,how='all')
                                            Labels=rawResults.columns[2:]
                                            Labels=[l for l in Labels if 'Unnamed' not in l]
                                            for D['label'] in Labels:
                                                LabelResults=DF(rawResults[D['label']])
                                                LabelResults.index=rawResults.T.iloc[0].values
                                                LabelScores=LabelResults[D['label']].loc[LearningScores]
                                                LabelScores=concat([LabelScores,DF(index=WeightsScores)])
                                                LabelWeights=[w for w in WeightsScores if w in LabelResults.index]
                                                for w in LabelWeights:
                                                    LabelScores.loc[w]=LabelResults[D['label']].loc[w]

                                                #save
                                                resultsDetails=DF.from_dict(D,orient='index')
                                                ResultsDF=concat([resultsDetails,LabelScores]).T
                                                FullResults=concat([FullResults,ResultsDF],ignore_index=True)
                                                FoundFilesCount+=1
                                        except IOError:
                                            resultsDetails=DF.from_dict(D,orient='index')
                                            missingFiles=concat([missingFiles,resultsDetails],axis=1,ignore_index=True)
                                            MissingFilesCount+=1
    saveName='PatientsVsControls-WithWeights'#raw_input('enter file save name: ')
    FullResults.to_csv(resultsDir+'FullResults_'+saveName+'.csv')
    missingFiles.to_csv(resultsDir+'FullResults_'+saveName+'-MissingFiles.csv')
    print('\nSUMMARY - found ' + str(FoundFilesCount) + ' out of ' + str(MissingFilesCount+FoundFilesCount)+' files' )
    print('successfully saved as : ' + resultsDir+'FullResults_'+saveName+'.csv')
        

summarizeAllResultsToCsv()