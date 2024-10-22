import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

#
#
#
# Load the Excel file
file_path = 'C:/Users/Prachi/Documents/Data Science Masters/Thesis/Klausurdaten.xlsx'
data = pd.read_excel(file_path)

# Extracting roll numbers and question scores
score_columns = [col for col in data.columns if 'Punkte' in col]
student_data = data[['Matr.-Nr.'] + score_columns]

# Get a list of all columns in the 'data' DataFrame
all_columns = data.columns.tolist()

# Remove the columns in 'score_columns' from this list
columns_to_keep_1 = [col for col in all_columns if col not in score_columns]
columns_to_keep = [col for col in columns_to_keep_1 if col != "Nummer" or col != "Scan-Nr."]

# Create a new DataFrame with the remaining columns
student_answer_class_data = data[columns_to_keep]

data.info()

#
#
#
# Data cleaning
# Convert score columns to numeric, handling any non-numeric entries
for col in score_columns:
    student_data[col] = pd.to_numeric(student_data[col].astype(str).str.replace(',', '.').str.replace(' ', ''), errors='coerce')

# Fill missing values with zero (assuming no score or missing data means zero score)
student_data.fillna(0, inplace=True)

# Normalize the data
scaler = StandardScaler()
student_data_scaled = scaler.fit_transform(student_data[score_columns])

# Display the first few rows of the preprocessed data
print(student_data.head())
print(student_data_scaled[:5])  # Show first few rows of both original and scaled data

# Adjusting the data preprocessing to handle non-string columns

# Convert score columns to strings and then to numeric, handling any non-numeric entries
for col in score_columns:
    student_data[col] = pd.to_numeric(student_data[col].astype(str).str.replace(',', '.').str.replace(' ', ''), errors='coerce')

# Fill missing values with zero (assuming no score or missing data means zero score)
student_data.fillna(0, inplace=True)


#
#
#
# Applying K-means clustering

# Normalize the data
scaler = StandardScaler()
student_data_scaled = scaler.fit_transform(student_data[score_columns])

# Display the first few rows of the preprocessed data
#print(student_data.head())
#print(student_data_scaled[:5])  # Show first few rows of both original and scaled data

# Elbow Method to find the optimal number of clusters
sum_of_squared_distances = []
K = range(1, 15)  # Trying different numbers of clusters (from 1 to 14)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(student_data_scaled)
    sum_of_squared_distances.append(km.inertia_)

# Plotting the elbow graph
plt.figure(figsize=(10, 6))
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')
plt.show()
'''
# Applying K-Means clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(student_data_scaled)

# Adding the cluster information to the original data
performance_data_with_clusters = student_data_scaled.copy()
performance_data_with_clusters['Cluster'] = clusters

# Display the first few rows of the data with cluster labels
performance_data_with_clusters.head()

'''