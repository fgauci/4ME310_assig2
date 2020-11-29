# **********************************************************************************************************************
# IMPORTING LIBRARIES
# **********************************************************************************************************************
import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

#Libraries required for plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib import colors
from matplotlib import figure
from matplotlib.colors import ListedColormap

#Library required for normalisation of data
from sklearn import preprocessing

#Libraries required for tuning parameters
from yellowbrick.cluster import KElbowVisualizer 
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

import itertools 

#Libraries required for clustering algorithms
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA

import warnings
warnings.simplefilter("ignore")

from matplotlib import style
style.use('seaborn')

# **********************************************************************************************************************
# DATA EXPLORATION
# **********************************************************************************************************************

#Source of dataset
#P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis, "Modeling wine preferences by data mining from physicochemical properties", Decision Support Systems, vol. 47, no. 4, pp. 547-553, 2009. Available: 10.1016/j.dss.2009.05.016.

#Importing the data files and setting delimiter to ';' 
#Adding categorical variable according to the type of wine

df_red=pd.read_csv("winequality-red.csv", delimiter=";")
df_red['type']='red'

df_white=pd.read_csv("winequality-white.csv", delimiter=";")
df_white['type']='white'

df=df_red.append(df_white)

def type_to_num(i):
    if i == 'red':
        return 1
    if i == 'white':
        return 2

df.to_csv('winequality.csv')

df['type_num']=df['type'].apply(type_to_num)
print(df)

#Listing features 
print()
print('-'*30 + '\nFEATURES\n' + '-'*30)
print(df.info())

#Checking for Null values
print()
print('-'*30 + '\nNULL VALUES\n' + '-'*30)
print(df.isnull().sum())

#Checking for Null values
print()
print('-'*30 + '\nNULL VALUES\n' + '-'*30)
print(df.isnull().sum())

#Checking for Duplicate values
print()
print('-'*30 + '\nDUPLICATE VALUES\n' + '-'*30)
print(df.duplicated())
print(df.duplicated().sum())

df_unique=df.drop_duplicates()

#Quality count 
f, axes=plt.subplots(1, 2)
f.set_size_inches(15, 7)

quality_count=sns.countplot(df_unique['quality'], color='orange', ax=axes[0])
quality_count.title.set_text('Quality Count')
quality_count.title.set_size(15)
for p in quality_count.patches:
    quality_count.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)

quality_density=sns.distplot(df['quality'], color='steelblue', ax=axes[1])
quality_density.title.set_text('Quality Density')
quality_density.title.set_size(15)

plt.subplots_adjust(wspace=0.3)
plt.show()

#Type count 
type_count=sns.countplot(df['type'], palette={'red':'r', 'white':'navajowhite'})
df['type'].value_counts()
for p in type_count.patches:
    type_count.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)
plt.title('Count of Wines by Type',fontsize=12)
sns.set(rc={'figure.figsize':(10,7.5)})
plt.show()

#------------------------
#PEARSON'S CORRELATION
#------------------------

#Correlation of features with quality
correlations=df_unique.corr()['quality'].sort_values(ascending=False)
correlations.plot(kind='bar',color='steelblue')
plt.tight_layout()
plt.title('Correlation of Features with Quality',fontsize=12)
plt.show()

#Correlation of features with type
correlations=df_unique.corr()['type_num'].sort_values(ascending=False)
correlations.plot(kind='bar',color='steelblue')
plt.tight_layout()
plt.title('Correlation of Features with Type',fontsize=12)
plt.show()

#Correlation Matrix
sns.heatmap(df_unique.corr(), cmap='Blues', annot=True, annot_kws={'size':12})
sns.set(rc={'figure.figsize':(30,20)})
sns.set(font_scale=2)
plt.tight_layout()
plt.title("Pearson's Correlation Matrix", fontsize=12) 
plt.show()

#------------------------
#BOXPLOTS
#------------------------

#Box plots for features correlated with type
bp, axes=plt.subplots(ncols=1, nrows=3,figsize=(16,30))
bp1=sns.boxplot(ax=axes[0],x='type',y='volatile acidity', data=df_unique, palette={'red':'r', 'white':'navajowhite'}).set(title="Alcohol in Different Types of Wine")
bp2=sns.boxplot(ax=axes[1],x='type',y='chlorides', data=df_unique, palette={'red':'r', 'white':'navajowhite'}).set(title="Sulphates in Different Types of Wine")
bp3=sns.boxplot(ax=axes[2],x='type',y='total sulfur dioxide', data=df_unique, palette={'red':'r', 'white':'navajowhite'}).set(title="Total Sulfur Dioxide in Different Types of Wine")
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.rc('font',size=8)
plt.show()

#Creating scatterplots of correlated features
f, axes=plt.subplots(ncols=1, nrows=3)
f1=sns.regplot(ax=axes[0], x='density',y='alcohol', data=df_unique, fit_reg=True, scatter_kws={'edgecolor':'w', 'linewidths':1}, color='steelblue').set(title="Alcohol and Density in Wine")
f2=sns.regplot(ax=axes[1], x='density',y='residual sugar', data=df_unique, fit_reg=True, scatter_kws={'edgecolor':'w', 'linewidths':1}, color='steelblue').set(title="Residual Sugar and Density in Wine")
f3=sns.regplot(ax=axes[2], x='total sulfur dioxide',y='free sulfur dioxide', data=df, fit_reg=True, scatter_kws={'edgecolor':'w', 'linewidths':1}, color='steelblue').set(title="Free Sulfur Dioxide and Total Sulfur Dioxide in Wine")
plt.subplots_adjust(wspace=0.3, hspace=0.4)
sns.set(rc={'figure.figsize':(10,15)})
plt.rc('font',size=8)
plt.show()

# **********************************************************************************************************************
# PREPROCESSING DATASET
# **********************************************************************************************************************

#Creating a new dataframe 'df_features' without the quality and type
df_features=df_unique.drop(['type', 'type_num', 'quality', 'density', 'total sulfur dioxide'], axis=1)
print(df_features)

#Normalising data in df_clustering and creating a dataframe df_normalised for the normalised data. This is going to be used for clustering.
features=df_features.values 
columns=df_features.columns
scaler=preprocessing.MinMaxScaler()
features_scaled=scaler.fit_transform(features)
df_normalised=pd.DataFrame(features_scaled, columns=columns)

#Plotting variance 
pca=PCA(n_components=9)
pca.fit(df_normalised)
variance=pca.explained_variance_ratio_ 
var=np.cumsum(np.round(variance, 3)*100)
plt.figure(figsize=(13.5,9))
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis',fontsize=15)
plt.ylim(0,100.5)
plt.plot(var)
plt.show()

#Principal component analysis (PCA) to reduce the dimensionality of the dataset to 2. 
pca=PCA(n_components=2)
pca_result=pca.fit_transform(df_normalised)
df_normalised['PCA1']=pca_result[:, 0]
df_normalised['PCA2']=pca_result[:,1]

df_pca=df_normalised[['PCA1', 'PCA2']].copy()

print()
print('-'*30 + '\nTOP 5 ROWS OF PCA DATA\n' + '-'*30)
print(df_pca.head())

# **********************************************************************************************************************
# KMEANS
# **********************************************************************************************************************

#Elbow method to find the optimal value of number of centroids k
model=KMeans()
visualizer=KElbowVisualizer(model, k=(1,10), timings=False, size=(900, 600))
visualizer.fit(df_pca)
visualizer.show()       
optimal_k=visualizer.elbow_value_

#K-means at k at optimal_k and maximum iterations set to 100
k_means =KMeans(init='k-means++', n_clusters=optimal_k, max_iter=100, random_state=0)
k_means.fit(df_pca)
centroids=k_means.cluster_centers_
kmeans_labels=k_means.labels_

#plotting KMeans results
colours=['#AC19E1', '#1980E1','#E19519']
custom_cmap=colors.ListedColormap(colours)
colours2=['#AC19E1', '#1980E1','#E19519','#7cb063','#d16e5a','#d174c1']
custom_cmap2=colors.ListedColormap(colours2)

#Scatter plot for K-means
plt.scatter(df_pca.iloc[:,0], df_pca.iloc[:,1], c=k_means.labels_, cmap=custom_cmap, edgecolors='w', linewidths=0.5, s=40)
plt.gcf().set_size_inches(10, 10)
plt.scatter(centroids[:,0], centroids[:,1], s=150, cmap=custom_cmap, marker='o', edgecolors='k', linewidths=1, c=np.unique(k_means.labels_))
plt.title('KMeans optimised with Sum of Squares (max iter=100)',fontsize=15)
plt.xlabel('PCA1', fontsize=12)
plt.ylabel('PCA2', fontsize=12)

#setting up the legend colors and labels
#get the unique labels
cmap=cm.get_cmap(custom_cmap)
unique_labels=np.unique(k_means.labels_)  
norm=colors.Normalize(vmin=0, vmax=(optimal_k-1))
class_colours=[cmap(norm(x)) for x in unique_labels]
recs=[]
for i in range(0,len(class_colours)):
    recs.append(mpatches.Rectangle((0,0),1,1, fc=class_colours[i]))
plt.legend(recs,unique_labels,loc=1)
plt.show()

# **********************************************************************************************************************
# DBSCAN
# **********************************************************************************************************************

#DBSCAN Method 1

#Taking [minPts=2*dim] and [k=2*dim - 1]. as suggested by Schubert et. al based on previous work by Sander et. al. 
dim=len(df_pca.columns)
minPts=2 * dim
kneighbours=minPts - 1

#Building K-distance graph to find optimal epsilon value and using kneed to get the exact value
nearest_neighbors=NearestNeighbors(n_neighbors=kneighbours)
neighbors=nearest_neighbors.fit(df_pca)
distances, indices=neighbors.kneighbors(df_pca)
distances=np.sort(distances[:,(kneighbours-1)], axis=0)
i=np.arange(len(distances))
knee=KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
knee.plot_knee(figsize=[13.5,9])
plt.xlabel("Points")
plt.ylabel("Distance")
plt.title('K-distance Graph',fontsize=15)
optimal_eps=distances[knee.knee]
plt.show()

print('min_samples=' + str(minPts))
print('n_neighbours=' + str(kneighbours))
print('eps=' + str(optimal_eps))

dbscan=DBSCAN(eps=optimal_eps, min_samples=minPts)
dbscan.fit(df_pca)
dbscan_labels=dbscan.labels_

silhouette=silhouette_score(df_pca,dbscan.labels_)
print(str('Silhouette Score: ' + str(silhouette)))

print('Number of Clusters: ' +str(len(np.unique(dbscan_labels))))

#DBSCAN Method 2

#Basing DBSCAN parameters on optimal silhouette score
for i, j in itertools.product(np.arange(0.001,0.05,0.005), range(4,30)):
    dbscan_sil=DBSCAN(eps=i, min_samples=j)
    dbscan_sil.fit(df_pca)
    dbscan_sil_labels=dbscan_sil.labels_
    if len(np.unique(dbscan_sil_labels)) > 1:
        silhouette=silhouette_score(df_pca,dbscan_sil.labels_)
        print('eps: ' + str(round(i,4)) +' & minPts: ' + str(j) + ', silhouette score: ' + str(silhouette) + ' & clusters: ', len(np.unique(dbscan_sil_labels)))

#Optimal silhouette score is 0.4768589816 at eps=0.046 and min_samples=9
dbscan2=DBSCAN(eps=0.046, min_samples=9)
dbscan2.fit(df_pca)
dbscan2_labels=dbscan2.labels_

#plotting DBSCAN results
df_pca['dbscan_label']=dbscan_labels
df_pca['dbscan2_label']=dbscan2_labels

f, axes=plt.subplots(ncols=2, nrows=1)
f.set_size_inches(15, 7)
dbscan1=sns.scatterplot(ax=axes[0], x='PCA1',y='PCA2', data=df_pca, hue='dbscan_label', palette=custom_cmap2, s=20)
dbscan1.title.set_text('DBSCAN optimised with K-distance (eps=0.011, min_samples=4, clusters=56)')

dbscan2=sns.scatterplot(ax=axes[1], x='PCA1',y='PCA2', data=df_pca, hue='dbscan2_label', palette=custom_cmap, s=20)
dbscan2.title.set_text('DBSCAN optimised with Silhoutte Score (eps=0.046, min_samples=9, clusters=2)')
plt.show()

#linear model plot for DBSCAN method 1, since the scatterplot combined multiple clusters into one 
#(e.g. clusters 1-10 into cluster 10)
cmap2= ['#AC19E1','#1980E1','#E19519','#7CB063','#D16E5a','#D174C1']
sns.lmplot(x='PCA1',y='PCA2', data=df_pca, fit_reg=False, hue='dbscan_label', palette=sns.color_palette(cmap2, n_colors=56),scatter_kws={'edgecolor':'w', 'linewidths':1, 's':50}, legend=False)
plt.title("DBSCAN optimised with K-distance (eps=0.011, min_samples=4, clusters=56)",fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.5, 0.5), ncol=1)
plt.show()

# **********************************************************************************************************************
# PLOTTING
# **********************************************************************************************************************

#adding labels to original dataframe to plot scatter plots of features 
df_unique['kmeans_label']=kmeans_labels
df_unique['dbscan_label']=dbscan_labels
df_unique['dbscan2_label']=dbscan2_labels

#Pairplot for KMeans (k=3,max_iter=100)
cmap1=['#AC19E1', '#1980E1','#E19519']
sns.pairplot(df_unique[df_unique.columns.difference(['dbscan_label','dbscan2_label','type','type_num','quality'])],hue='kmeans_label', palette=cmap1)
sns.set(rc={'figure.figsize':(50,50)})
plt.show()

#Pairplot for DBSCAN Method 1 (eps and min_samples based on Schubert et. al)
cmap2= ['#AC19E1','#1980E1','#E19519','#7CB063','#D16E5a','#D174C1']
sns.pairplot(df_unique[df_unique.columns.difference(['kmeans_label','dbscan2_label','type','type_num','quality'])],hue='dbscan_label', palette=sns.color_palette(cmap2, n_colors=56))
plt.show()

#Pairplot for DBSCAN Method 2 (eps and min_samples based on Silhoutte Score)
cmap1=['#AC19E1', '#1980E1','#E19519']
sns.pairplot(df_unique[df_unique.columns.difference(['kmeans_label','dbscan_label','type','type_num','quality'])],hue='dbscan2_label', palette=sns.color_palette(cmap1, n_colors=2))
plt.show()

# **********************************************************************************************************************
# ANALYSIS OF RESULTS
# **********************************************************************************************************************

df.to_csv('winequality_clustered.csv', index=False)
 
#Number of points in each KMeans Cluster
kmeans_clusters_count=sns.countplot(df_unique['kmeans_label'],color='orange')
df_unique['kmeans_label'].value_counts()
for p in kmeans_clusters_count.patches:
    kmeans_clusters_count.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)
plt.title('KMeans Clusters Count',fontsize=15)
plt.show()

#Number of points in each DBSCAN Cluster for Method 1
dbscan_clusters_count=sns.countplot(df_unique['dbscan_label'],color='orange')
df_unique['dbscan_label'].value_counts()
for p in dbscan_clusters_count.patches:
       dbscan_clusters_count.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)
plt.title('DBSCAN (Method 1) Cluster Count',fontsize=15)
plt.show()

#Number of points in each DBSCAN Cluster for Method 2
dbscan2_clusters_count=sns.countplot(df_unique['dbscan2_label'],color='orange')
df_unique['dbscan2_label'].value_counts()
for p in dbscan2_clusters_count.patches:
       dbscan2_clusters_count.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)
plt.title('DBSCAN (Method 2) Cluster Count',fontsize=15)
plt.show()

#Distribution of Quality within the KMeans Cluster
kmeans_quality_count=sns.countplot(x='kmeans_label',hue='quality',data=df_unique,palette='Blues_r')
for p in kmeans_quality_count.patches:
       kmeans_quality_count.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)
plt.legend(loc='upper right', ncol=1,title="quality")
sns.set(rc={'figure.figsize':(12,10)})
plt.title('Quality in KMeans Clusters',fontsize=15)
plt.show()

#Distribution of Type within the KMeans Cluster
f, axes = plt.subplots(1)
f.set_size_inches(15, 7)
kmeans_quality_count = sns.countplot(x='kmeans_label', hue='type', data=df_unique,palette={'red':'r', 'white':'navajowhite'})
for p in kmeans_quality_count.patches:
       kmeans_quality_count.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=8)
sns.set(rc={'figure.figsize':(6,5)})
plt.title('Type in KMeans Clusters',fontsize=10)
plt.show()

#Further analysis to understand KMeans clusters. 
sns.lmplot(size=6, x='density',y='alcohol', data=df_unique, fit_reg=False, hue='kmeans_label', palette=sns.color_palette(cmap2, n_colors=56),scatter_kws={'edgecolor':'w', 'linewidths':1, 's':50})
sns.set(rc={'figure.dpi':150})
plt.title("KMeans Clusters for Alcohol and Density",fontsize=12)
plt.xlim(0.98,1.02)
plt.show()

sns.lmplot(size=7, x='total sulfur dioxide',y='free sulfur dioxide', data=df_unique, fit_reg=False, hue='kmeans_label', palette=sns.color_palette(cmap2, n_colors=56),scatter_kws={'edgecolor':'w', 'linewidths':1, 's':50})
plt.title("KMeans Clusters for Free Sulfur Dioxide and Total Sulfur Dioxide",fontsize=12)
plt.xlim(0,400)
plt.show()

sns.lmplot(size=7, x='total sulfur dioxide',y='free sulfur dioxide', data=df_unique, fit_reg=False, hue='kmeans_label', palette=sns.color_palette(cmap2, n_colors=56),scatter_kws={'edgecolor':'w', 'linewidths':1, 's':50})
plt.title("KMeans Clusters for Free Sulfur Dioxide and Total Sulfur Dioxide",fontsize=12)
plt.xlim(0,400)
plt.show()