#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:30:23 2017

@author: dburlins
"""
import numpy
import csv
import os
import sys
from numpy import linalg


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

######################################################
###
###     Math functions
######################################################

# # # # 
# Find the mean value of a list of numbers
def findMean( X ): 
    sum = 0;
    for x in X:
        sum += x
    return sum / len(X)


# # # 
# Find the variance of a list of numbers
def findVariance( X ):
    var = 0
    mean = findMean( X )

    for x in X:
        var += ( x - mean ) ** 2

    return var / len(X)


# # #
# Find the standard deviation of a list of numbers
def findStdDev( X ):
    return findVariance( X ) ** ( 0.5 ) 


# # #
# Mean-center the given list of numbers
def meanCenter( X ):
    mean = findMean( X )
    for i in xrange(len(X)): 
        X[i] = X[i] - mean
    return X


# # #
# Find the covariance of X and Y 
#   E(XY)E(X)E(Y)
def findCovariance( X, Y ):
    XY = numpy.array(X) * numpy.array(Y)
    return findMean(XY) - findMean(X) * findMean(Y)


######################################################
###
###     Read data from a csv file 
######################################################
def readFile( filename ):
    loc = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(loc, filename), 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        for row in data:
            try:
                X.append(float(row[0]))
                Y.append(float(row[1]))
                Z.append(float(row[2]))
            except:
                continue   
            

######################################################
######################################################
######################################################


# Declare variable lists
X = []
Y = []
Z = []

# Populate variables from csv
readFile("quizdata.csv")


# Mean Center each variable
meanCenter(X)
meanCenter(Y)
meanCenter(Z)



# print the variance for each variable
print 'variance of X:', findVariance(X)
print 'variance of Y:', findVariance(Y)
print 'variance of Z:', findVariance(Z)



# print the covariance between variables
print 'covariance of X, Y:', findCovariance(X,Y)
print 'covariance of Y, Z:', findCovariance(Y,Z)



# plot the original data in 3D space
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d') 
ax.plot(X, Y, Z, 'o', markersize=4, color='blue', alpha=0.5)
plt.title('Samples from X Y Z')
plt.xlabel('x scores')
plt.ylabel('y scores')
plt.show()


# Calculate the covariance matrix
cov = numpy.cov([X, Y, Z])


# Calculate the eigenvalues and eigenvectors of the covariance matrix
# Keep them in sorted order (largest to smallest eigenvalue)
tmp = zip(*sorted(zip(linalg.eig(cov)[0],linalg.eig(cov)[1].T),reverse=True))
eigenvalues = tmp[0]
P = numpy.array(tmp[1]) #eigenvectors corresponding to ordered eigenvalues


# new coordinate system 
# vectors of newX are new principal components
newX = P.dot([X,Y,Z])

# shows that the variance of each new principal component 
#   is equivalent to the corresponding eigenvalue
#print findVariance(newX[0]), eigenvalues[0]
#print findVariance(newX[1]), eigenvalues[1]
#print findVariance(newX[2]), eigenvalues[2]


# new matrix w to go from 3 to 2 dimensions 
w = numpy.hstack((P[0].reshape(3,1), P[1].reshape(3,1)))
xformed = w.T.dot([X,Y,Z])

# Plot the lower dimensional space
plt.plot(xformed[0,:], xformed[1,:], 'o', markersize=4, color='blue', alpha=0.5, label='')
plt.xlabel('x scores')
plt.ylabel('y scores')
plt.title('Transformed samples')
plt.show() 





# question 3
A = [[0, -1], [2, 3]]
linalg.eig(A)