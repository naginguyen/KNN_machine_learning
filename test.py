# data preprocessing
import pandas as pd
import csv
from sklearn.utils import shuffle

# check if there is any duplicates

df = pd.read_csv('data 2.csv')
duplicates = df[df.duplicated(keep= False)]

if not duplicates.empty:
    print("Duplicated rows found:")
    print(duplicates)

df_cleaned = df.drop_duplicates()
print(f"Number of rows after removing duplicates: {len(df_cleaned)}")

# relabel the dataset
df_cleaned['popularity'] = pd.qcut(df_cleaned['popularity'], q=4, labels=[0, 1, 2, 3])

# shuffle
# frac=1: chọn ngẫu nhiên tất cả các hàng- 100%
df_shuffled = df_cleaned.sample(frac=1).reset_index(drop=True)

# export dataset into a new csv file
df_shuffled.to_csv("shuffled_dataset.csv", index=False)

import pandas as pd
import csv
from sklearn.model_selection import train_test_split


data= pd.read_csv('shuffled_dataset.csv')
#train= train2+ val
#20% for test set
train, test= train_test_split(data, test_size= 0.2)
#use .head() to visualize in a table + an int as parameter

#train2 is the real training dataset that we use
#val is for validation
#10% for validation set
train2, val= train_test_split(train, test_size= 0.1)
train2.to_csv("FinalTrain_set.csv")
val.to_csv("FinalVal_set.csv")
test.to_csv("FinalTest_set.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('shuffled_dataset.csv')
X = data[['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
          'liveness', 'loudness', 'speechiness', 'tempo', 'explicit', 'key', 'mode']]
n_samples, n_features = X.shape
y = data['popularity']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)
real_eigenvalues = pca.explained_variance_


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_features+1), real_eigenvalues, marker='o')
plt.title('Scree Plot with Parallel Analysis (Manual)')
plt.xlabel('Component Number')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data= pd.read_csv('shuffled_dataset.csv')
#only choose numeric columns (not categorial)
numeric_cols= data.select_dtypes(include=['number'])

correlation= numeric_cols.corr().round(2)
sns.heatmap(correlation, annot = True)
plt.figure(figsize=(100,100))  # Phóng to kích thước hình ảnh
plt.show()


# picks top k features based on a scoring function
from sklearn.feature_selection import SelectKBest
# scoring function = ANOVA F-test to find the features most related to the target variable
from sklearn.feature_selection import f_classif
import pandas as pd

# Load data
data = pd.read_csv('shuffled_dataset.csv')

# Only choose numeric columns (not categorical)
X = data[['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
          'liveness', 'loudness', 'speechiness', 'tempo']]
y = data['popularity']

# Apply SelectKBest
# create SelectKBest using f_classif
# k=4: top 4 features most related to y
fvalue_Best = SelectKBest(f_classif, k= 5)
# fit: computes F score between each feature in X & target y
# transform: keep only top 3 scoring features
X_kbest = fvalue_Best.fit_transform(X, y)

f_scores= fvalue_Best.scores_
# Get selected feature names by applying X.columns
# fvalue_Best.get_support(): return boolean mask indicating which features were selected by SelectKBest
selected_features = X.columns[fvalue_Best.get_support()]

# Print results
# .shape[1]: returns the number of columns
for feature, score in zip(X.columns, f_scores):
    print(f"{feature}: {score:.4f}")

print("Selected feature names:", selected_features.tolist())
print("Original number of features:", X.shape[1])
print("Reduced number of features:", X_kbest.shape[1])



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load training data
training = pd.read_csv("FinalTrain_set.csv", usecols=['name', 'popularity', 'energy', 'loudness', 'year','acousticness','instrumentalness'])
training_array = training[['energy', 'loudness', 'year','acousticness','instrumentalness']].to_numpy()
category_array = training['popularity'].to_numpy()


#Load validating data
validating= pd.read_csv("FinalVal_set.csv", usecols= ['name', 'popularity', 'energy', 'loudness', 'year','acousticness','instrumentalness'])
validating_array= validating[['energy', 'loudness', 'year','acousticness','instrumentalness']].to_numpy()

def minkowski_distance(x, y, p= 5):
    sum=0
    for i in range(len(x)):
        sum += abs(x[i] - y[i])**p
    return sum ** (1 / p)

# K-Nearest Neighbors (KNN) function
# point: STT của observation trong validateSet
# k: k nearest neighbors

def KNN (trainSet, trainLabels, validateSet, point, k):
    distances = []

    # Calculate distances between the point and all training data
    for i in range(len(trainSet)):
        dist = minkowski_distance(trainSet[i], validateSet[point])
        #add distances to the list
        distances.append((dist, i, trainLabels[i]))

    # Sort distances in descending order and select top k
    distances.sort()
    top_k_distances = distances[:k]


    # Display the song details
    val_row= validating.iloc[point]
    print(f"The selected song: {val_row['name']}- Year: {val_row['year']}, Energy: {val_row['energy']}, "
    f"Loudness: {val_row['loudness']}, Acousticness: {val_row['acousticness']}, Instrumentalness: {val_row['instrumentalness']}, Popularity score: {val_row['popularity']}")

    print(f"First {k} neighbors: ")
    for i, (dist, idx, _) in enumerate(top_k_distances):
        row = training.iloc[idx]  # Get song details from DataFrame
        print(f"{i}. {row['name']}- Year: {row['year']}, Energy: {row['energy']}, "
        f"Loudness: {row['loudness']}, Acousticness: {row['acousticness']}, Instrumentalness: {row['instrumentalness']}, Popularity score: {row['popularity']}, Distance: {dist}")

    weights_count = {0: 0, 1: 0, 2: 0, 3: 0}
    weights_sum = {0: 0, 1: 0, 2: 0, 3: 0}
    weights_average= {0: 0, 1: 0, 2: 0, 3: 0}

    for dist, _,  category in top_k_distances:
        if dist==0:
            weight= float('inf') # no authority in voting
        else:
            weight= 1/dist
        weights_sum[category] += weight
        weights_count[category] += 1

    for key in weights_average:
        if weights_count[key] != 0:
            weights_average[key]= weights_sum[key]/weights_count[key]

    print(f"Predicted popularity: {max (weights_average, key= weights_average.get)}")


#Scale the features for better performance
scaler= StandardScaler()
training_array_scaled= scaler.fit_transform(training_array)
validating_array_scaled= scaler.transform(validating_array)

KNN(training_array_scaled, category_array, validating_array_scaled, 200, 350)

# fix with weighted knn


import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

# Load training data
training = pd.read_csv("FinalTrain_set.csv", usecols=['name', 'popularity', 'year', 'energy', 'acousticness', 'instrumentalness', 'loudness'])
training_array = training[['year', 'energy', 'acousticness', 'instrumentalness', 'loudness']].to_numpy()
category_array = training['popularity'].to_numpy()


#Load validating data
validating= pd.read_csv("FinalVal_set.csv", usecols= ['name', 'popularity', 'year', 'energy', 'acousticness','instrumentalness','loudness'])
validating_array= validating[['year', 'energy', 'acousticness', 'instrumentalness', 'loudness']].to_numpy()
valcat_array= validating['popularity'].to_numpy()

#Load testing data
testing= pd.read_csv("FinalTest_set.csv", usecols= ['name', 'popularity', 'year', 'energy', 'acousticness','instrumentalness', 'loudness'])
test_array= testing[['year', 'energy', 'acousticness', 'instrumentalness', 'loudness']].to_numpy()
testcat_array= testing['popularity'].to_numpy()


# Scale the features for better performance
scaler= StandardScaler()
training_array_scaled= scaler.fit_transform(training_array)
validating_array_scaled= scaler.transform(validating_array)
testing_array_scaled= scaler.transform(test_array)

k= math.floor(math.sqrt(len(training)))
knn= KNeighborsClassifier(n_neighbors= k)
knn.fit(training_array_scaled, category_array)

predicted_val=[]
predicted_test=[]

for i in range (len(validating)):
    a= KNN(training_array_scaled, category_array, validating_array_scaled, i, 30)
    predicted_val.append(a)

for i in range (len(testing)):
    b= KNN(training_array_scaled, category_array, test_array_scaled, i, 30)
    predicted_test.append(b)

print("Accuracy score of the validating set: ", accuracy_score(valcat_array, predicted_val))
print("Accuracy score of the testing set: ", accuracy_score(testcat_array, predicted_test))



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load training data
training = pd.read_csv("FinalTrain_set.csv", usecols=['name', 'popularity', 'energy', 'loudness', 'year','acousticness','instrumentalness'])
training_array = training[['energy', 'loudness', 'year','acousticness','instrumentalness']].to_numpy()
category_array = training['popularity'].to_numpy()

#Load validating data
validating= pd.read_csv("FinalVal_set.csv", usecols= ['name', 'popularity', 'energy', 'loudness', 'year','acousticness','instrumentalness'])
validating_array= validating[['energy', 'loudness', 'year','acousticness','instrumentalness']].to_numpy()
valcat_array = validating['popularity'].to_numpy()

#Load testing data
testing= pd.read_csv("FinalTest_set.csv", usecols= ['name', 'popularity', 'year', 'energy', 'acousticness','instrumentalness', 'loudness'])
test_array= testing[['year', 'energy', 'acousticness', 'instrumentalness', 'loudness']].to_numpy()
testcat_array= testing['popularity'].to_numpy()

#Scale the features for better performance
scaler= StandardScaler()
training_array_scaled= scaler.fit_transform(training_array)
validating_array_scaled= scaler.transform(validating_array)
test_array_scaled= scaler.transform(test_array)

knn= KNeighborsClassifier(n_neighbors=350, weights= 'distance')
knn.fit(training_array_scaled, category_array)

predicted_val= knn.predict(validating_array_scaled)
predicted_test= knn.predict(test_array_scaled)

print("Accuracy score of the validating set: ", accuracy_score(valcat_array, predicted_val))
print("Accuracy score of the testing set: ", accuracy_score(testcat_array, predicted_test))