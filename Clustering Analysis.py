#!/usr/bin/env python
# coding: utf-8

# # Cluster Analysis (Hierarchical & K-Means)

# ## Introduction and Business Problem
# Compnay XYZ, a wholesales distributor operating in Portugal serves a diverse range of business clients, including retailer, restaurants, hotels and cafes. The marketing manager of XYZ want to understand more about their clients' spending patterns. By gaining insights, the goal of this anlaysis is to help XYZ company to tailor their strategies to satisfy clients' demand and manage inventory well. 

# ## Procedure of Analsis
# To find out clients' spending patterns, I implemented clustering analysis using python. I implemented hierarchical and k-means clustering in this dataset. This algorithm helps group similar data together into a specific number of clusters. After finding out specific clusters, I tried to understand differenct clusters of clents by exploring data anlaysis. Based on insights I gain, I provied suggested recommendations for XYZ company.

# ### Variable Name Description
# - Channel: Client channel (“1” means Horeca (Hotel/Restaurant/Cafe) and
# “2” means Retail)
# - Region: Client region (“1” means Lisbon, “2” means Oporto, and “3” means
# other regions)
# - Fresh: Annual spending on fresh products.
# - Milk: Annual spending on milk products.
# - Grocery: Annual spending on grocery products.
# - Frozen: Annual spending on frozen products.
# - Detergents Paper: Annual spending on detergents and paper products.
# - Delicatessen: Annual spending on deli products.

# ### Table of contents:
# 1. Problem defining and our overall rationale to solve it
# 2. EDA before Clustering
# 3. Hierarchical Clustering:
# to derive an initial idea of clustering number then conduct k-means clustering to check based on
# the evaluation of SSE curve, cluster plot, and Silhouette coefficient.
# 4. K-Means Clustering:
# to see if our cluster number is appropriate and reasonable to do further interpretation. If yes, the
# data with cluster label we use in the managerial document is from hierarchical clustering, since the
# cluster labels of k-means clustering are different each time (same pattern, different labels assigned)
# 5. Evaluate Clustering Solutions: SSE, cluster plot, and Slihouette Coefficient.
# 6. Analysis after clustering
# 7. Summary for clustering and other analysis results
# 8. Translate analyzing results into business solutions outline

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import MinMaxScaler


# ### Problem Define and Rationale to Solve the problem

# #### General goal for analysis
# To get a deeper understanding of the spending patterns of the clients.
# Specific goal for analysis:
# 1. conduct clustering analysis and discover different patterns of spending for each cluster.
# 2. we need to do EDA before moving on to further analysis, and we are going to combine
# analysis from that as well as clustering to provide data support for business solutions.
# 3. we will translate analyzing results into business solutions outline, and more detailed
# business strategies will be presented in our managerial document.

# In[2]:


wholesales = pd.read_csv("Wholesale customers data.csv")
wholesales.head()


# In[3]:


wholesales.describe()


# In[4]:


### There are 440 data points. XYZ company's clients average spent the most on Fresh Food


# In[5]:


# calculate the proportions for Region and Channel
region_count = wholesales["Region"].value_counts()
channel_count = wholesales["Channel"].value_counts()

# create a 1X2 grid for the pie charts with equal aspect ratio
plt.figure(figsize=(12,5))

# create a pie chart for Region
plt.subplot(1, 2, 1, aspect='equal')
plt.pie(region_count, labels = region_count.index, autopct= "%1.1f%%", colors= ["r","g","b"])
plt.title("Region Proportion")

# create a pie chart for Channel
plt.subplot(1,2,2, aspect = "equal")
plt.pie(channel_count, labels = channel_count.index, autopct= "%1.1f%%", colors= ["gold", "coral"])
plt.title("Channel Proportion")
plt.tight_layout()
plt.show()


# - Based on the pie chart, other regions contributed the most. However, if only take region 1 and 2 into consideration, XYZ company focus more on Lisbon region (region 1). Also, channel 1 (Horeca) is twice than channel 2 (Retail).

# In[6]:


## Check the correlation among variables to get a conceptual and general understanding of the dataset
corr_matrix = wholesales.iloc[:, 3:9].corr()
corr_matrix


# In[7]:


# create a correlation heatmap
plt.figure(figsize =(10,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# - From the plot above, we found no obvious negative correlation between variables. What we believe more worth highlighting is the relatively strong positive correlation between Grocery and Detergents_Paper, and then are Milk and Grocery, Milk and Detergent_Paper.

# In[8]:


# Group the data by Region and Channel
channel_grouped = wholesales[["Channel", "Milk", "Grocery","Frozen","Detergents_Paper","Delicassen"]].groupby("Channel").sum()
channel_grouped.head()


# In[9]:


# Create a bar chart
channel_grouped.plot(kind="bar", figsize=(10, 6))
plt.xlabel('Channel')
plt.ylabel('Total Spending')
plt.title('Total Spending by Product Category and Channel')
plt.legend(title='Product Category')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[10]:


# Group the data by Region and Channel
channel_grouped = wholesales[["Region", "Milk", "Grocery","Frozen","Detergents_Paper","Delicassen"]].groupby("Region").sum()
channel_grouped.head()


# In[11]:


# Create a bar chart
channel_grouped.plot(kind="bar", figsize=(10, 6))
plt.xlabel('Region')
plt.ylabel('Total Spending')
plt.title('Total Spending by Product Category and Region')
plt.legend(title='Product Category')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[12]:


# Group the data by Region and Channel
channel_mean = wholesales[["Channel", "Milk", "Grocery","Frozen","Detergents_Paper","Delicassen"]].groupby("Channel").mean()
channel_mean.head()


# In[13]:


# Create a bar chart
channel_mean.plot(kind="bar", figsize=(10, 6))
plt.xlabel('Channel')
plt.ylabel('Average Spending')
plt.title('Average Spending by Product Category and Channel')
plt.legend(title='Product Category')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[14]:


# Group the data by Region and Channel
region_mean = wholesales[["Region", "Milk", "Grocery","Frozen","Detergents_Paper","Delicassen"]].groupby("Region").mean()
region_mean.head()


# In[15]:


# Create a bar chart
region_mean.plot(kind="bar", figsize=(10, 6))
plt.xlabel('Region')
plt.ylabel('Average Spending')
plt.title('Average Spending by Product Category and Region')
plt.legend(title='Product Category')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ## Hierarchical Clustering
# ### Calculate Distances

# In[16]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler


# In[17]:


## Normalize the data using Min-Max normalization
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(wholesales.iloc[:, 3:9])
normalized_df = pd.DataFrame(normalized_data, columns= wholesales.columns[3:9])
normalized_df.head()


# In[18]:


## calculate the distance matrix using Euclian distance
from scipy.spatial.distance import pdist
distance_matrix = pdist(normalized_df, metric='euclidean')

## Perform hierarchical clustering using Ward's method
linkage_matrix = linkage(distance_matrix, method="ward")


# In[19]:


# Calculate the condensed distance matrix using Euclidean distance
from scipy.spatial.distance import pdist
distance_matrix = pdist(normalized_df, metric='euclidean')

# Perform hierarchical clustering using Ward's method
linkage_matrix = linkage(distance_matrix, method='ward')

# Plot the dendrogram without x-axis labels
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, no_labels=True)  # Set no_labels=True to hide x-axis labels
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('Distance')
plt.show()


# ## K-means
# 
# ### Use the k-means method to do the clustering analysis. I plotted a SSE curve to look into the proper number of clusters.

# In[20]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Select the features you want to use for clustering
features = wholesales[["Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]]

# Standardize the data (important for k-means)
scaler = StandardScaler()
kmeans = KMeans(n_init=10)

# Create a pipeline for standardization and k-means clustering
pipeline = make_pipeline(scaler, kmeans)

# Calculate SSE (Sum of Squared Errors) for different values of k
sse = []
for k in range(1, 11):
    pipeline.named_steps['kmeans'].n_clusters = k
    pipeline.fit(features)
    sse.append(pipeline.named_steps['kmeans'].inertia_)

# Plot the SSE curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.title('SSE Curve for K-Means Clustering')
plt.grid(True)
plt.show()


# ### From the SSE curve above, we could tell the elbow point is 4 or 5 clusters. Hence we need to further see how these two clusters work to cluster the dataset. First, we try 5 cluster because the slop of k=5 is less steep.

# In[21]:


from sklearn.decomposition import PCA

# Select the columns you want to use for clustering
selected_columns = wholesales[["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]]

# Create a StandardScaler instance
scaler = StandardScaler()

# Normalize the selected columns
normalized_data = scaler.fit_transform(selected_columns)

# Set the number of clusters to 5
k = 5

# Create and fit the K-means model
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
wholesales['cluster'] = kmeans.fit_predict(normalized_data)

# Apply PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_data)

# Visualize the clusters in 2D using PCA
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=wholesales['cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering Results (PCA)')
plt.colorbar(label='Cluster')
plt.show()


# In[22]:


# Set the number of clusters to 5
# Group the data by the 'cluster' column
cluster_grouped = wholesales.groupby('cluster')

# Iterate through each cluster and print the information
for cluster, group_data in cluster_grouped:
    num_data_points = len(group_data)
    avg_spending = group_data[["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]].mean()
    
    print(f"Cluster {cluster}:")
    print(f"Number of Data Points: {num_data_points}")
    print("Average Spending Across Categories:")
    print(avg_spending)
    print("\n")


# ### When clusters = 5, the cluster 4 would contain only 1 data point. This would not make any sense in real world explanation. Hence, I further try out 4 clusters to see if the clusters make more sense.

# In[23]:


# Set the number of clusters to 4
k = 4

# Create and fit the K-means model
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
wholesales['cluster'] = kmeans.fit_predict(normalized_data)

# Apply PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_data)

# Visualize the clusters in 2D using PCA
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=wholesales['cluster'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering Results (PCA)')
plt.colorbar(label='Cluster')
plt.show()


# In[24]:


# Set the number of clusters to 4
# Group the data by the 'cluster' column
cluster_grouped = wholesales.groupby('cluster')

# Iterate through each cluster and print the information
for cluster, group_data in cluster_grouped:
    num_data_points = len(group_data)
    avg_spending = group_data[["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]].mean()
    
    print(f"Cluster {cluster}:")
    print(f"Number of Data Points: {num_data_points}")
    print("Average Spending Across Categories:")
    print(avg_spending)
    print("\n")


# In[25]:


wholesales.head()


# ### After combined the insights from Hierarchical Clustering and K-means Clustering, I would pick 4 clusters and do further analysis based on these clusters.

# In[26]:


# Group the data by Region and Channel
cluster_total = wholesales[["cluster", "Milk", "Grocery","Frozen","Detergents_Paper","Delicassen"]].groupby("cluster").sum()
cluster_total.head()


# In[27]:


# Group the data by Cluster and calculate the total spending
cluster_total.plot(kind="bar", figsize=(10, 6))
plt.xlabel('Cluster')
plt.ylabel('Total Spending')
plt.title('Total Spending by Product Category and Region')
plt.legend(title='Product Category')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[28]:


# Group the data by Cluster and calculate the mean
cluster_avg = wholesales[["cluster", "Milk", "Grocery","Frozen","Detergents_Paper","Delicassen"]].groupby("cluster").mean()
# Create a bar chart
cluster_avg.plot(kind="bar", figsize=(10, 6))
plt.xlabel('Cluster')
plt.ylabel('Average Spending')
plt.title('Average Spending by Product Category and Region')
plt.legend(title='Product Category')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[29]:


cluster_avg


# ### Based on two bar chart above, we ccould tell that though cluster 1 and 2 spent more in total, especially in Milk and Grocery product. However, when it comes to average spending, cluster 3 and 5 spend higher because the data point is relatively less than other clusters. From this insights, we could infer that cluster 3 and 4 would be more potential clients that XYZ company could focus on. As a recommendations, I would suggest to do bundle deal promotion to boost Grocery and Milk sales together.

# In[30]:


# Create a bar chart showing region by clusters
plt.figure(figsize=(10, 6))
sns.countplot(data=wholesales, x='cluster', hue='Region')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Clusters by Region')
plt.legend(title='Region')
plt.show()


# In[31]:


# Create a bar chart showing region by clusters
plt.figure(figsize=(10, 6))
sns.countplot(data=wholesales, x='cluster', hue='Channel')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Clusters by Channel')
plt.legend(title='Channel')
plt.show()


# ### From region, all clusters seems to have similar patterns, which spend more on others region followed by Lisbon region.
# ### From channel, we would find out interesting pattern that cluster 1 focus more on Retail and clsuter two focus more on Horea.
# ### Hence, I would recommend that XYZ company should do inventory management based on different clsuters demands.

# In[32]:


# Export the data to a CSV file with cluster information
wholesales.to_csv("wholesales_with_clusters.csv", index=False)

