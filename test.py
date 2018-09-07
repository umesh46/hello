# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:25:10 2017

@author: umesh.k
"""
import numpy as np

sizes = [2,3,1]

biases = [np.random.randn(y, 1) for y in sizes[1:]]

weights = [np.random.randn(y, x)   for x, y in zip(sizes[:-1], sizes[1:])]


print(biases)
print(weights)
print(sizes[1:])
print(sizes[:-1],sizes[1:])

for b, w in zip(biases, weights):
    print(b,w)

    