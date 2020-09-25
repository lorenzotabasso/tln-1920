from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import collections

# np array con i valori di similarietà
test = [0., 0.70412415, 0.20412415, 0.41493811, 0.57305199, 0.15811388, 0.21320072, 0.21320072, 0., 0.,0.23570226, 0.73570226, 0., 0.5, 0., 0., 0.5, 0.81622777, 0.31622777, 0.23570226, 0.5029635 , 0.45624348, 0.68898224, 0.65811388, 0.15811388, 0.5, 0.5, 0.31725282, 0.56725282, 0.57203059, 0.32203059, 0.5, 0., 0.72237479, 0.22237479, 0.31725282, 0.54085962, 0.2236068 , 0.17149859, 0.52505198, 0.35355339, 0., 0.2409996 , 0.4819992 , 0.66225829, 0.42125869, 0., 0., 0.19611614, 0.36761472, 0.17149859, 0., 0.20412415, 0.70412415, 0.5, 0., 0., 0., 0.43952454, 0.59028021, 0.55900396, 0.90824829, 0., 0., 0.5, 0.5, 0.5, 0.16666667, 0.40236893, 0.23570226, 0., 0.33419801, 0.33419801, 0., 0.31622777, 0.81622777, 0.75, 0.47301681, 0.72277262, 0.49975581, 0.31622777, 0.65156306, 0.3353353 , 0.25, 0.49618298, 0.24618298, 0.23570226, 0.48188524, 0.24618298, 0.19611614, 0.19611614, 0.26726124, 0.26726124, 0., 0.36324158, 0.36324158, 0.31990258, 0.31990258, 0.21320072, 0.40218295, 0.18898224, 0.20412415, 0.40024028, 0.19611614, 0., 0., 0.16222142, 0.32444284, 0.42948266, 0.26726124, 0., 0., 0.5, 0.65430335, 0.3086067 , 0.15430335, 0., 0.28867513, 0.72876374, 0.]
similarities = np.array(test)
data = similarities.reshape(-1, 1)

# Use silhouette score to find optimal number of clusters to segment the data
num_clusters = np.arange(2,10)
results = {}
for size in num_clusters:
    model = KMeans(n_clusters = size).fit(data)
    predictions = model.predict(data)
    results[size] = silhouette_score(data, predictions)

best_size = max(results, key=results.get)
print(best_size)

kmeans = KMeans(n_clusters = 3)
kmeans.fit(data)
matix_clusterized = kmeans.labels_
print(matix_clusterized)

count_matrix = collections.Counter(matix_clusterized)
print(count_matrix)

first_chars = len(matix_clusterized) / best_size
print(first_chars)

count_0 = 0
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0

# TODO: scorrere la finestra con la sliding window
for i in range(int(first_chars)):
    if matix_clusterized[i] == 0:
        count_0 += 1
    if matix_clusterized[i] == 1:
        count_1 += 1
    if matix_clusterized[i] == 2:
        count_2 += 1
    if matix_clusterized[i] == 3:
        count_3 += 1
    if matix_clusterized[i] == 4:
        count_4 += 1

print(count_0,count_1,count_2,count_3,count_4)