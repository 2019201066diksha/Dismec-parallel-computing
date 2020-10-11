#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:11:40 2020

@author: diksha
"""

import numpy as np
import pandas as pd
from mpi4py import MPI
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
x_train = None
num_labels = None
y_train_small = None
y_train = None
if(rank == 0):
    print(rank,"executing")
    with open('EURLex-4K/trn_ft_mat.txt') as file:
      content = file.readlines()
    content =[x.strip() for x in content]
    datapoints,featurecount = content[0].split(' ')    
    datapoints = int(datapoints)
    featurecount  = int(featurecount)
    trdataframe = np.zeros(shape=(datapoints+1,featurecount+1))
    #print("hii")
    for i in range(len(content)):
        if i==0:
            continue
        else:
            line = content[i]
            line = line.split(' ')
            for j in line:
                feature_id,feature_value = j.split(':')
                trdataframe[i][int(feature_id)] = float(feature_value)
    #print("hii")
    with open('EURLex-4K/trn_lbl_mat.txt') as file:
        content = file.readlines()
    content =[x.strip() for x in content]
    datapoints,labelcount = content[0].split(' ')    
    datapoints = int(datapoints)
    labelcount  = int(labelcount)    
    lbldataframe = np.full((datapoints+1,labelcount+1),-1)
    #print("hii")
    for i in range(len(content)):
        if i==0:
            continue
        else:
            line = content[i]
            line = line.split(' ')
            for j in line:
              if(len(j)>0):
                  label_id = j.split(':')
                  label_col = label_id[0]
                  label_val = label_id[1]
                  # print(type(label_val))
                  lbldataframe[i][int(label_col)] = int(label_val)
                  # print(lbldataframe[i][int(label_col)])
    #print('hii')
    with open('EURLex-4K/tst_ft_mat.txt') as file:
        content = file.readlines()
    content =[x.strip() for x in content]
    datapoints,ftcount = content[0].split(' ')    
    datapoints = int(datapoints)
    ftcount  = int(ftcount)    
    tstdataframe = np.zeros(shape=(datapoints+1,ftcount+1))
    
    for i in range(len(content)):
        if i==0:
            continue
        else:
            line = content[i]
            line = line.split(' ')
            for j in line:
              if(len(j)>0):
                  label_id = j.split(':')
                  label_col = label_id[0]
                  label_val = label_id[1]
                  # print(type(label_val))
                  tstdataframe[i][int(label_col)] = float(label_val)
    #print("hii")
    trDataframe = np.delete(trdataframe,obj=0,axis=0)
    lblDataframe = np.delete(lbldataframe,obj=0,axis=0)
    tstDataframe = np.delete(tstdataframe,obj=0,axis=0)
    
    trDataframe = np.delete(trdataframe,obj=0,axis=1)
    lblDataframe = np.delete(lbldataframe,obj=0,axis=1)
    tstDataframe = np.delete(tstdataframe,obj=0,axis=1)

#     print("hii")
#   
    #print(type(trDataframe))
    x_train,x_test,y_train,y_test = train_test_split(trDataframe,lblDataframe, test_size = 0.3,random_state=69)
    print(x_train.shape,y_train.shape,rank)
    #print('hii')
    #comm.send((x_train,y_train[0,1:10]),1)
    #x_train = comm.bcast(x_train, root=0)
    #data = comm.recv(source=0)
    #print(data[0].shape,data[1].shape)
    #pass
    #data = comm.recv(source=0)
    #print(type(data))808
    y_train_small = np.array((y_train[:,:]))
    print("y_trainsmall shape : ",y_train_small.shape)
    num_labels = y_train_small.shape[1]
    print("y_train dtype",y_train.dtype)

x_train = comm.bcast(x_train, root=0)
num_labels = comm.bcast(num_labels, root=0)
labels_per_rank = int(num_labels/size)
if(rank!=0):
    print(x_train.shape,rank)
start,end=[],[]
for r in range(size):
    s = r*labels_per_rank
    if(r != size-1):
        e = s+labels_per_rank-1
    else:
        e = num_labels-1
    start.append(s)
    end.append(e)

if(rank == 0):
    #print(start,end)
    y_train_subset = y_train_small[:,start[0]:end[0]+1]
    for i in range(1,size):
        comm.send(y_train_small[:,start[i]:end[i]+1],i)
else:
    y_train_subset = comm.recv(source=0)
    
W = sparse.csr_matrix(np.zeros((x_train.shape[1],y_train_subset.shape[1])))

for col in range(y_train_subset.shape[1]):
    clf = LinearSVC()
    if( np.unique(y_train_subset[:,col] ).shape[0]>1  ):  
        clf.fit(x_train,y_train_subset[:,col])
        wvec = clf.coef_[0]
        wvec = np.where(abs(wvec)>0.01,wvec,0)
    else:
        wvec = np.zeros(x_train.shape[1])
    wvec = sparse.csr_matrix(wvec)
    W[:,col] = wvec.T
if(rank == 0):
    for i in range(1,size):
        returned_w = comm.recv(source=i)
        W = hstack([W,returned_w])
        print("shape of W:",W.shape)
    print("---------------------------------------------------------")
    #lala = np.array([1 if W[.dot(x_test[j,:])>0 else -1 for j in range(x_test.shape[0])])
    pred = x_test@W
    check_for_label = 2
    p = [1 if i > 0 else -1 for i in pred[:,check_for_label]]
    print(accuracy_score(y_test[:,check_for_label],p))
    
else:
    comm.send(W,0)
    

#if(rank != size-1):
#    start = rank*labels_per_rank
#    end = start+labels_per_rank-1
#    print('for rank',rank,'start :',start,"end :",end)   
#else:
#    start = rank*labels_per_rank
#    end = num_labels-1
#    print('for rank',rank,'start :',start,"end :",end)
    
#y_train_subset = np.empty(shape=[x_train.shape[0],end-start+1],dtype=np.int64)
#print("rank",rank,"subset",y_train_subset.shape)
#comm.Scatter(y_train_small,y_train_subset, root=0)
#print(y_train_subset[0])

#comm.scatter(y_train,y_train_subset,)   





                                        
