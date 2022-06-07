import pandas as pd
import numpy as np
import math
import operator


#### Start of STEP 1
# Importing data 
data = pd.read_csv('Train.csv')

data1 = pd.read_csv('Train1.csv')
#### End of STEP 1


# Defining a function which calculates euclidean distance between two data points
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

# Defining our KNN model
def knn(trainingSet,trainingSet2 ,testInstance, k):
 
    distances = {}
    sort = {}
 
    length = testInstance.shape[0]
    
    #### Start of STEP 3
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        
        #### Start of STEP 3.1
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)

        distances[x] = dist
        #### End of STEP 3.1
 
    #### Start of STEP 3.2
    # Sorting them on the basis of distance
    sorted_d = (sorted(distances.items(), key =
             lambda kv:(kv[1], kv[0])))
    #### End of STEP 3.2
 
    neighbors = []
    # print (distances)
    # print ('/n,/n/,/n')
    #print (sorted_d)
    
    #### Start of STEP 3.3
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x+2][0])
    #### End of STEP 3.3
    classVotes = {}
    
    #### Start of STEP 3.4
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = trainingSet2.iloc[neighbors[x]][-1]
 
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #### End of STEP 3.4

    #### Start of STEP 3.5
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0], neighbors)
    #### End of STEP 3.5


test = pd.read_csv('Test.csv')

#### Start of STEP 2
# Setting number of neighbors = 1

k = 4
#### End of STEP 2
# Running KNN model
for x in range(len(test)):
    result,neigh = knn(data,data1, test.iloc[x], k)
    print (result)
