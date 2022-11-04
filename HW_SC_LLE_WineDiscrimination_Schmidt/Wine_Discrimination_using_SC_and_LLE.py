# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 20:14:07 2022

@author: Alina Schmidt
"""
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import SpectralClustering 
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.decomposition import PCA 


# Spectral Clustering

wine_red = pd.read_csv("winequality-red.csv", sep=";")
wine_white = pd.read_csv("winequality-white.csv",sep=";")

wine = pd.concat([wine_red, wine_white])

# Scale and normalize
scaler = StandardScaler() 
wine_norm = normalize(scaler.fit_transform(wine))

wine_norm = pd.DataFrame(wine_norm)

# Create a vector with colors distinguishing the two wine sorts.
def colors(i):
    if i<=1599:
        i = 'red'
    else:
        i = 'grey'
    return(i)

wine_color = [colors(i) for i in range(6497)]



# Reducing the dimensions of the data 
pca = PCA(n_components = 2) 
wine_pca = pca.fit_transform(wine_norm)  

X_principal = pd.DataFrame(wine_pca)
X_principal.columns = ['P1', 'P2'] 


fig = plt.figure(figsize=(10, 10))
plt.scatter(X_principal['P1'], X_principal["P2"],c=wine_color, marker='o')
plt.savefig("PCA_Red_and_White_Wine.png")
# Red wine can be seen in upper left corner, but a few points also among 
# cluster of white wine.


spectral_model_rbf = SpectralClustering(n_clusters = 2, affinity ='rbf') 
labels_rbf = spectral_model_rbf.fit_predict(wine_norm)

# Visualizing the clustering 
plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = SpectralClustering(n_clusters = 2, affinity ='rbf') .fit_predict(X_principal), cmap =plt.cm.autumn) 
plt.show() 
plt.savefig("SpectralClustering_Wine.png")
# The spectral clustering does not show much distance between the two groups.




# LLE 

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold, datasets



X_r, err = manifold.locally_linear_embedding(wine_norm, n_neighbors=10,
                                             n_components=2)

fig = plt.figure(figsize=(10, 10))
plt.scatter(X_r[:,0], X_r[:,1], marker='o', c=wine_color, cmap=plt.cm.Spectral)

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Reduced to 2D using LLE')
plt.savefig("LLE_Wine.png")
# Here, LLE distinguishes red and white wine worse than PCA.
