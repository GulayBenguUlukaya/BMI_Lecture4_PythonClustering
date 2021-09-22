#import guppy
#from guppy import hpy
#heap = hpy()
#heap_status1 = heap.heap()

# Import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing
from matplotlib import pyplot as plt #plotting figures

df = pd.read_csv("./dataset/Iris.csv") #load the dataset
x = df.iloc[:, [1, 2, 3, 4]].values #drop categorical values

#Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
fig=plt.figure()

#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Within cluster sum of squares') 

plt.draw()
plt.savefig('./output/iris_kmeans3_elbowplot.png', dpi=300, bbox_inches='tight')

###Time model training
#%%time
#Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
#Wall time: 20.9 ms

fig=plt.figure()

#Visualizing the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 30, c = 'orange', label = 'K Means Cluster 0')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 30, c = 'blue', label = 'K Means Cluster 1')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 30, c = 'green', label = 'K Means Cluster 2')

#Plotting the centers of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Cluster Centers',marker='*')

plt.legend()

plt.draw()
plt.savefig('./output/iris_kmeans3_clusters.png', dpi=300, bbox_inches='tight')

actual = df["Species"].tolist() #Convert actual species to list

#Assign species to generated clusters, create predicted species list to compare with actual 
pred = ['Iris-setosa' if (i==1) else 'Iris-versicolor' if (i==0)  else 'Iris-virginica' if (i==2) else i for i in y_kmeans] 

from sklearn.metrics import confusion_matrix
#Calculate confusion matric between actual species and predicted by k-means clustering
cm = confusion_matrix(actual, pred)

import seaborn as sns; sns.set()

x_axis_labels = ['Iris-setosa','Iris-versicolor','Iris-virginica'] # labels for x-axis
y_axis_labels = ['K Means Cluster 1','K Means Cluster 0','K Means Cluster 2'] # labels for y-axis

#Plot confusion matrix
fig=plt.figure()

ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=x_axis_labels, yticklabels=y_axis_labels)

plt.draw()
plt.savefig('./output/iris_kmeans3_confusionMatrix.png', dpi=300, bbox_inches='tight')

import csv
#Output k-means-predicted species
with open('./output/iris_kmeans3_predictions.csv', 'w') as f:
    writer = csv.writer(f)
    for val in pred:
        writer.writerow([val])

#Calculate accuracy of k-means clustering results and output
from sklearn.metrics import accuracy_score
acc = accuracy_score(actual, pred, normalize=True, sample_weight=None)
        
print("K Means Clustering applied to Iris dataset with "+str(acc)+" accuracy of predicting species.")
#heap_status2 = heap.heap()
#print("\nMemory Usage After Creation Of Objects : ", heap_status2.size - heap_status1.size, " bytes")
#Memory Usage After Creation Of Objects :  71068128  bytes
