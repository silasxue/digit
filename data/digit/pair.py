# -*- coding: utf-8 -*-
from os import listdir
from random import shuffle
from scipy.io import savemat
import numpy as np
ratio = 0.1
train_txt = open('train.txt', 'w')
test_txt = open('test.txt', 'w')
test_list = []
train_list = []

SHUFFLE = 0

for i in range(11):
    imgs = listdir(str(i))
    shuffle(imgs)
    nimgs = len(imgs)
    ntest = int(nimgs * ratio)
    ntrain = int(nimgs - ntest)
    print str(nimgs)+' '+str(i)
    for j in range(0, ntrain):
        if SHUFFLE:        
            train_list.append(str(i) + '/' + imgs[j] + ' ' + str(i) + '\n')
        else:
            train_txt.write(str(i) + '/' + imgs[j] + ' ' + str(i) + '\n')
    for j in range(ntrain, nimgs):
        test_txt.write(str(i) + '/' + imgs[j] + ' ' + str(i) + '\n')
        test_list.append(str(i) + '/' + imgs[j])

if SHUFFLE:
    shuffle(train_list)
    for i in train_list:
        train_txt.write(i)

ntest = len(test_list)
test_imgs = np.zeros((ntest,), dtype=np.object)

for i in range(ntest):
    test_imgs[i] = test_list[i]
savemat('test.mat',{'pair': test_imgs});

train_txt.close()
test_txt.close()