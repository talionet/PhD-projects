#Load Data From Excel and from MATLAB
from pandas import DataFrame as DF
from pandas import *
import xlrd
import pickle
xlsDataPath='C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Matlab\\Data_Handle\\'
AllAUs= ['au1','au2','au3','au4','au5','au6','au7','au8','au9','au10','au11','au12','au13','au14','au15','au16','au17','au18','au19','au20','au21','au22','au23','au24','au25','au26','au27','au28','au29','au30','au31','au32','au33','au34','au35','au36','au37','au38','au39','au40','au41','au42','au43','au44','au45','au46','au47','au48']
book=xlrd(xlsDataPath)
AllSubjectsDF=DF()
for subject in subjectsList:
    print(subject)
    sheet=book.sheet_by_name(subject)
    subjectList=[]
    rowRange=range(sheet.nrows)
    multiInd=MultiIndex.from_product([subject,rowRange])
    multiInd.names=['subject','time']
    for row in range(sheet.nrows):
        subjectList.append(sheet.row_values(row))
    subjectDF=DF(subjectList,index=multiInd)
    AllSubjectsDF=concat([AllSubjectsDF,subjectDF])
AllSubjectsDF.columns=AllAUs
AllSubjectsDF.to_csv("C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\LearningData\\photosDATAraw.csv")
savefile=open("C:\\Users\\taliat01\\Desktop\\TALIA\\Code-Python\\Results\\LearningData\\photosDATAraw.csv",'wb')
pickle.dump(AllSubjectsDF,savefile)