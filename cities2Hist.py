#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:04:17 2017

@author: deaxman
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
wordDict={}
with open('testCities.txt') as f:
    for line in f:
        for word in line.split(','):
            if word.rstrip().strip() in wordDict:
                wordDict[word.rstrip().strip()]+=1
            else:
                wordDict[word.rstrip().strip()]=1
print(sum(wordDict.values()))
valList=sorted(wordDict.values())
sortIdx=[i[0] for i in sorted(enumerate(wordDict.values()), key=lambda x:x[1])]
keyList=[list(wordDict.keys())[i] for i in sortIdx]
plt.bar(range(len(keyList)),valList)
plt.xticks(range(len(keyList)), keyList,rotation='vertical')
#%%
import codecs 
with codecs.open('1.txt','r',encoding='utf-8') as f1:
    lines=[str(l) for l in f1.readlines()]
for i in range(45,117):
    with open(str(i),'wb') as f:
        [f.write(line.encode("utf8")) for line in lines]
        
#%%
def flatten(l):
    full=[]
    for e in l:
        if type(e) == list:
            full=full+flatten(e)
        else:
            full=full+[e]
    return full
#%%
searchResults=[]
for i in range(1,77):
    with codecs.open('/Users/deaxman/Documents/Job Search_2017/Positions/' + str(i)+'.txt','r',encoding='utf-8') as f1:
        if 'news' in [w.rstrip().strip() for w in flatten([l.split() for l in f1.readlines()])]:
            searchResults.append(i)
print(searchResults)
#%%
import pandas as pd
import matplotlib.pyplot as plt
import re
import os.path
import codecs
wordDict={}
for i in range(1,123):
    if os.path.isfile('/Users/deaxman/Documents/Job Search_2017/Notes/'+str(i)+'.txt'):
        with codecs.open('/Users/deaxman/Documents/Job Search_2017/Notes/'+str(i)+'.txt','r',encoding='utf-8') as f:
            qualFlag=False
            for line in f:
                if qualFlag:
                    for word in re.split('-| ',line):
                        if word.rstrip().strip() in wordDict:
                            wordDict[word.rstrip().strip()]+=1
                        else:
                            wordDict[word.rstrip().strip()]=1
                if line=='Qualifications:\n':
                    qualFlag=True
    else:
        with codecs.open('/Users/deaxman/Documents/Job Search_2017/Notes/'+str(i),'r',encoding='utf-8') as f:
            qualFlag=False
            for line in f:
                if qualFlag:
                    for word in re.split('-| ',line):
                        if word.rstrip().strip().lower() in wordDict:
                            wordDict[word.rstrip().strip().lower()]+=1
                        else:
                            wordDict[word.rstrip().strip().lower()]=1
                if line=='Qualifications:\n':
                    qualFlag=True

wordDict2={}
for key in wordDict.keys():
    if wordDict[key]>10:
        wordDict2[key]=wordDict[key]
valList=sorted(wordDict2.values())
sortIdx=[i[0] for i in sorted(enumerate(wordDict2.values()), key=lambda x:x[1])]
keyList=[list(wordDict2.keys())[i] for i in sortIdx]
plt.bar(range(len(keyList)),valList)
plt.xticks(range(len(keyList)), keyList,rotation='vertical')


#%%
from scipy.stats import binom
print(binom.pmf(6,10,0.5))

#%%
def combineLists(a,b,fullList):
    if len(a)==0:
        fullList.append(b[0])
    elif len(b)==0:
        fullList.append(a[0]) 
    elif a[0]>b[0]:
        fullList.append(b[0])
        combineLists(a,b[1:],fullList)
    else:
        fullList.append(a[0])
        combineLists(a[1:],b,fullList)
        
#%%
def combineLists2(a,b):
    if len(a)==0:
        return b 
    if len(b)==0:
        return a 
    if a[0]>b[0]:
        return [b[0]]+combineLists2(a,b[1:])
    else:
        return [a[0]]+combineLists2(a[1:],b)
    
#%%
def isPalindrome(s):
    if len(s)>2:
        return (s[0]==s[-1]) and isPalindrome(s[1:-1])
    elif len(s)==2:
        return (s[0]==s[-1])
    else:
        return True
        
#%%
def printDupes(strIn):
    d={}
    for s in strIn:
        if s not in d.keys():
            d[s]=1
        else:
            d[s]+=1
    print([i for i in d.keys() if d[i]>1])

def printDupes2(strIn):
    d=[]
    for i,s in zip(range(len(strIn)),strIn):
        if s in list(strIn)[i+1:]:
            d.append(s)
    print(d)
        
        
def unique(trends):
    print(list(set(trends)))

def binSearch(elem,arr):
    mid=arr[len(arr)//2]
    if mid==elem:
        return True
    elif elem>mid:
        return binSearch(elem,arr[(len(arr)//2)+1:])
    else:
        return binSearch(elem,arr[:(len(arr)//2)])
    return False


def topDownView(root):
    output=[root.getValue()]
    current=root
    while current.getRight() != None:
        current=current.getRight()
        output.append(current.getValue())
    current=root
    while current.getRight() != None:
        current=current.getRight()
        output.insert(0,current.getValue())
        
        
        
class Node():
    def __init__(self, value,left=None,right=None):
        self.value=value
        self.left=left
        self.right=right
    def setLeft(self,node):
        self.left=node
    def setRight(self,node):
        self.right=node
    
import numpy as np
def printGap(num):
    numList=list(str(num))
    numLen=len(numList)
    for groupSize in range(1,numLen//2):
        tmp=[numList[i2:i2+groupSize] for i2 in range(0,numLen,groupSize)]
        groupedNumList=np.array([int(i) for i in tmp])
        print(groupedNumList)
        numListIncrements=np.diff(groupedNumList)  
        if (np.sum(numListIncrements==1)==(len(numListIncrements)-1)) and np.sum(numListIncrements==2)==1:
            print(int(np.argwhere(numListIncrements==2)+1))
            return groupedNumList[int(np.argwhere(numListIncrements==2)+1)]-1
    return -1
                
            
def findPairsWithSum(numList,sumVal):
    hashTable={}
    output=[]
    for i in range(sumVal-max(numList),sumVal-min(numList)+1):
        hashTable[i]=0
    for i1 in numList:
        hashTable[i1]=1
    for i2 in numList:
        if hashTable[sumVal-i2]==1:
            output.append([i2,sumVal-i2])
    return output
            
def findAnagrams(wordList,dictList):
    output={}
    for word in wordList:
        output[word]=[]
        for word2 in dictList:
            if sorted(list(word))==sorted(list(word2)):
                print((list(word)).sort())
                print((list(word2)).sort())
                output[word].append(word2)
    print(output)
    
    
#%%
from scipy.misc import *
import requests
from io import BytesIO
import pandas as pd
import numpy as np
import urllib
import time
import re
def getNumPopInRange(zipCodeStr,ageRange):
    response = requests.get('http://pics4.city-data.com/zag/za' + zipCodeStr + '.png')
    im = imread(BytesIO(response.content))
    rows,cols,clr =im.shape
    ageGraph=[]
    ageGraph2=[]
    for col in range(cols):
        tmpRed=-1
        tmpPink=-1
        for row in range(285):
            if np.sum(im[row,col,:]==np.array([160,0,0]))==3:
                tmpRed=285-row
            if np.sum(im[row,col,:]==np.array([255,0,255]))==3:
                tmpPink=285-row
        if not (tmpRed==-1 and tmpPink==-1):
            if tmpPink!=-1:
                ageGraph.append(tmpPink)
            else:
                ageGraph.append(tmpRed)
            if tmpRed!=-1:
                ageGraph2.append(tmpRed)
            else:
                ageGraph2.append(tmpPink)
    #return [np.sum(ageGraph[int(ageRange[0]*5.7):int(ageRange[1]*5.7)]),np.sum(ageGraph2[int(ageRange[0]*5.7):int(ageRange[1]*5.7)]),np.sum(ageGraph)+np.sum(ageGraph2)]
    ageRank=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,5,5,6,12,13,13,13,10,8,7,6,4,3,3,3,2,2,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    return np.dot(np.array([np.sum(ageGraph[int(i*5.7):int(5.7*(i+1))])/(np.sum(ageGraph)+np.sum(ageGraph2)) for i in range(85)]),(np.array(ageRank/np.sum(ageRank))))
    #return [np.sum(ageGraph[int(ageRange[0]*5.7):int(ageRange[1]*5.7)])/(np.sum(ageGraph[int(20*5.7):int(55*5.7)])+np.sum(ageGraph2[int(20*5.7):int(55*5.7)]))]


#%% GET


df=pd.read_excel('Zipcode-ZCTA-Population-Density-And-Area-Unsorted.xlsx',names=['zip', 'pop', 'land', 'density'])
zipdf=df[df['density']>8000]
zipdf=zipdf[zipdf['pop']>5000]
girlDen=[]
girlPercent=[]
girlToBoyRatio=[]
zipDict={}
for i in zipdf.iterrows():
    print(i[1]['zip'])
    if i[1]['zip']<10000:
        try:
            tmp='0'+str(int(i[1]['zip']))
            popVal=getNumPopInRange(tmp,[20,26])
        except:
            popVal=[0,0,1]
    else:
        try:
            tmp=str(int(i[1]['zip']))
            popVal=getNumPopInRange(tmp,[20,26])
        except:
            popVal=[0,0,1]
    popDensity=i[1]['density']
    zipDict[tmp]=popVal
    if popVal[1]!=0:
        girlToBoyRatio.append(popVal[0]/popVal[1])
    else:
        girlToBoyRatio.append(0)
    girlDen.append(popDensity*(popVal[0]/popVal[2]))
    girlPercent.append(popVal[0]/popVal[2])
    
    
zipdf['GirlRatio']=pd.Series(girlToBoyRatio,index=zipdf.index.values)
zipdf['GirlDensity']=pd.Series(girlDen,index=zipdf.index.values)
zipdf['GirlPercent']=pd.Series(girlPercent,index=zipdf.index.values)
zipdf.to_excel('LivableZip.xlsx','Sheet1')

#%% FILTER
zipdf2=zipdf[(zipdf['GirlPercent']>.15) & (zipdf['GirlRatio']>0.9) & (zipdf['GirlDensity']>2000)]
zipdf2.to_excel('LivableZip2.xlsx','Sheet1')

#%%  GET loc info
girlPercent=[]
gymNum=[]
restaurantNum=[] 
yogaNum=[]
barNum=[]
cafeNum=[]
for i in zipdf2.iterrows():
    print(i[0])
    if int(i[1]['zip'])<10000:
        tmp='0'+str(int(i[1]['zip']))
    else:
        tmp=str(int(i[1]['zip']))
    gymNum.append(float(findPlaces('gyms',tmp)))
    restaurantNum.append(float(findPlaces('restaurants',tmp)))
    yogaNum.append(float(findPlaces('yoga',tmp)))
    time.sleep(5)
    barNum.append(float(findPlaces('bar',tmp)))
    cafeNum.append(float(findPlaces('cafe+and+coffee+wifi',tmp)))
    time.sleep(5)
zipdf2['gymNum']=pd.Series(gymNum,index=zipdf2.index.values)
zipdf2['restaurantNum']=pd.Series(restaurantNum,index=zipdf2.index.values)
zipdf2['yogaNum']=pd.Series(yogaNum,index=zipdf2.index.values)
zipdf2['barNum']=pd.Series(barNum,index=zipdf2.index.values)
zipdf2['cafeNum']=pd.Series(cafeNum,index=zipdf2.index.values)

zipdf2['gymNum']=pd.to_numeric(zipdf2['gymNum'], errors='coerce')
zipdf2['restaurantNum']=pd.to_numeric(zipdf2['restaurantNum'], errors='coerce')
zipdf2['yogaNum']=pd.to_numeric(zipdf2['yogaNum'], errors='coerce')
zipdf2['barNum']=pd.to_numeric(zipdf2['barNum'], errors='coerce')
zipdf2['cafeNum']=pd.to_numeric(zipdf2['cafeNum'], errors='coerce')



zipdf2['gymPer']=zipdf2['gymNum']/zipdf2['land']
zipdf2['restaurantPer']=zipdf2['restaurantNum']/zipdf2['land']
zipdf2['yogaPer']=zipdf2['yogaNum']/zipdf2['land']
zipdf2['barPer']=zipdf2['barNum']/zipdf2['land']
zipdf2['cafePer']=zipdf2['cafeNum']/zipdf2['land']

zipdf2.to_excel('LivableZip2.xlsx','Sheet1')
#%%FILTER by locinfo

zipdf5=zipdf2[zipdf2['gymPer']>50]
zipdf5=zipdf2[zipdf2['restaurantPer']>400]
zipdf5=zipdf2[zipdf2['yogaPer']>20]
zipdf5=zipdf2[zipdf2['barPer']>200]
zipdf5=zipdf2[zipdf2['cafePer']>80]

zipdf5.to_excel('LivableZip3.xlsx','Sheet1')


#%%
def findPlaces(place,zipCode):
    from bs4 import BeautifulSoup
    html = urllib.request.urlopen('https://www.yelp.com/search?find_desc='+place+'&find_loc='+zipCode).read()
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)
    matchObj = re.findall( r'Showing 1-\d+ of \d+', str(html), re.M|re.I)
    return matchObj[0].split(' ')[-1]
#%%
import time
def testYelp(tmp):
    gymNum=[]
    restaurantNum=[] 
    yogaNum=[]
    barNum=[]
    cafeNum=[]
    gymNum.append(findPlaces('gyms',tmp)) 
    restaurantNum.append(findPlaces('restaurants',tmp))
    yogaNum.append(findPlaces('yoga',tmp))
    time.sleep(5)
    barNum.append(findPlaces('bar',tmp))
    cafeNum.append(findPlaces('cafe+and+coffee+wifi',tmp))
    time.sleep(5)
    
#%%
def findMaxSubseq(seq,minSize=1,maxSize=None):
    intSeq=np.cumsum(seq)
    if minSize==1 and maxSize==None:
        minIdx=0
        bestRange=[0,0]
        maxSum=0
        for idx,elem in enumerate(intSeq):
            if intSeq[minIdx]>elem:
                minIdx=idx
            if elem-intSeq[minIdx]>maxSum:
                maxSum=elem-intSeq[minIdx]
                bestRange=[minIdx+1,idx]
            
    else:
        tmpMax=-100
        print('dustin')
        for idx,elem in enumerate(intSeq[0:(-maxSize)]):
            print(idx)
            if tmpMax<(np.max(intSeq[idx+minSize:idx+maxSize])-elem):
                tmpMax=(np.max(intSeq[idx+minSize:idx+maxSize])-elem)
                bestRange=[idx+1,idx+np.argmax(intSeq[idx+minSize:idx+maxSize])]
        

    return bestRange



