#Dataframe manipulation library
import pandas as pd
#Math functions, we'll only need the sqrt function so let's import only that
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
#get_ipython().run_line_magic('matplotlib', 'inline')
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import json

data_loc= "/home/shubham/Desktop/Function_App_Testing/HttpTrigger1/Guest Recommender.csv"


df=pd.read_csv(data_loc)

pd.set_option('display.max_columns', None)

df['ARRDAT'] = pd.to_datetime(df['ARRDAT'], format='%Y%m%d')

df['DEPDAT'] = pd.to_datetime(df['DEPDAT'], format='%Y%m%d')

df['RMNTS']=df['DEPDAT']-df['ARRDAT']

df['RMNTS']=df['RMNTS'].astype('timedelta64[D]')

df.drop(['REGNUB','FOLNUB','ROMNUB','ARRDAT','DEPDAT','TRC','COM','CHQ','TRD','CAS','ADQ','CRD','ADC','BOH','ADV','POT'],axis=1,inplace=True)

df[['TRS', 'PHT', 'SPC', 'BBD', 'GBR', 'JVH', 'MIT', 'EP', 'VCH', 'RNT', 'ITV', 'CCF', 'PHC', 'TIP', 'EXB', 'STD', 'ART', 'SEC', 'NOT', 'FAX', 'FST', 'BB', 'RMS', 'ITB', 'HTM', 'TRF', 'RTN', 'LCA', 'PLT', 'PLC', 'BTB', 'FNC', 'SSG', 'LLT', 'LAU', 'IDD', 'JBR', 'MIS', 'AP', 'FX']]=df[['TRS', 'PHT', 'SPC', 'BBD', 'GBR', 'JVH', 'MIT', 'EP', 'VCH', 'RNT', 'ITV', 'CCF', 'PHC', 'TIP', 'EXB', 'STD', 'ART', 'SEC', 'NOT', 'FAX', 'FST', 'BB', 'RMS', 'ITB', 'HTM', 'TRF', 'RTN', 'LCA', 'PLT', 'PLC', 'BTB', 'FNC', 'SSG', 'LLT', 'LAU', 'IDD', 'JBR', 'MIS', 'AP', 'FX']].div(df.RMNTS, axis=0)

df.loc[df['RMNTS']==0]

df.isnull().sum() # This is to confirm that division by 0 has caused the NaN

df.dropna(axis=0,how='any', inplace=True) # if any value in the row is NaN, it will be removed. Else use how='all'

df.drop(['SPC'],axis=1,inplace=True)

df['BBD'].describe()

df.drop(['BBD'],axis=1,inplace=True)


df.drop(['GBR'],axis=1,inplace=True)

# #### Find all columns of the dataframe that have all 0 values in it. (It would be better to drop these in one go, than one by one).

zeros=df.loc[:,(df==0).all()] # 17 columns of the dataframe are fully 0s

df.drop(columns=zeros,axis=1,inplace=True) # zeros was assigned columns of the df in the previous cell


mms=MinMaxScaler([0,5]) # The parameter passed the range of values min=0 and max=5

df[['PHT','JVH','MIT','EP','VCH','EXB','ART','SEC','FAX','BB','RMS','ITB','HTM','TRF','RTN','BTB','FNC','LAU','MIS','RMNTS']]=mms.fit_transform(df[['PHT','JVH','MIT','EP','VCH','EXB','ART','SEC','FAX','BB','RMS','ITB','HTM','TRF','RTN','BTB','FNC','LAU','MIS','RMNTS']])

df=df.round(decimals=2)

df=df.drop(['GSTNAM'], axis=1) #Dropping Guest names so as to avoid confusions as there are several guests with same name
print(df.columns)
def recommend(gstcod):
    df[df['GSTCOD']==gstcod]
    #Grouping by guest code and returning the mean of rest of the columns
    df1 = df.groupby(['GSTCOD']).mean()
    df1.reset_index(inplace=True)
    df2 = df1.melt(id_vars=['GSTCOD'],var_name='Services',value_name='Rating')
    inputuser = df2[df2['GSTCOD']==gstcod]
    # inputuser[inputuser['Rating']!=0]
    # except_inputuser = df2[df2['GSTCOD']!=gstcod]
    # # except_inputuser
    # usersubset = except_inputuser[except_inputuser['Services'].isin(inputuser['Services'].tolist())][except_inputuser['Rating']!=0]
    # userSubsetGroup = usersubset.groupby(['GSTCOD'])
    # userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)
    # userSubsetGroup = userSubsetGroup[0:100]
    # pearsonCorrelationDict = {}
    # for name, group in userSubsetGroup:
    #     #Let's start by sorting the input and current user group so the values aren't mixed up later on
    #     group = group.sort_values(by='Services')
    #     inputuser = inputuser.sort_values(by='Services')
    #     #Get the N for the formula
    #     nRatings = len(group)
    #     #Get the ratings for the services that they both have in common
    #     temp_df = inputuser[inputuser['Services'].isin(group['Services'].tolist())]
    #     #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    #     tempRatingList = temp_df['Rating'].tolist()
    #     #Let's also put the current user group ratings in a list format
    #     tempGroupList = group['Rating'].tolist()
    #     #Now let's calculate the pearson correlation between two users, so called, x and y
    #     Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    #     Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    #     Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    #     #If the denominator is different than zero, then divide, else, 0 correlation.
    #     if Sxx != 0 and Syy != 0:
    #         pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    #     else:
    #         pearsonCorrelationDict[name] = 0
        
    # pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    # pearsonDF = pearsonDF.reset_index()
    # print(pearsonDF.head())
    # pearsonDF = pearsonDF.rename(columns = {'index': 'GSTCOD', 0: 'similarityIndex'}, inplace = False)
    # print(pearsonDF.head())
    
    
    # topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
    # except_inputuser1=except_inputuser[except_inputuser['Rating']!=0]
    # topUsersRating=topUsers.merge(except_inputuser1, left_on='GSTCOD', right_on='GSTCOD', how='inner')

    # topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['Rating']
    # tempTopUsersRating = topUsersRating.groupby('Services').sum()[['similarityIndex','weightedRating']]
    # tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
    # tempTopUsersRating.head()

    # recommendation_df = pd.DataFrame()
    # recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
    # recommendation_df['Services'] = tempTopUsersRating.index
    # recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
    # recommendList=recommendation_df.values.tolist()
    # print(recommendList)
    return inputuser.shape

g=int(input("Enter GSTCOD: "))#print(type (g))recommend(g)
print(recommend(g))
