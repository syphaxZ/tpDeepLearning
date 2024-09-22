#!/usr/bin/env python
# coding: utf-8

# # PARTIE 1 : Préparation des Données 

# 1.1 Importation des bibliothèques nécessaires

# In[1]:


import numpy as np


# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets 
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[7]:


from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage 
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


# 1.2 Chargement des jeux de données

# In[8]:


iris = datasets.load_iris()


# In[9]:


wine = datasets.load_wine()


# In[10]:


wholesale = pd.read_csv('Wholesale customers data.csv')


# In[11]:


iris.target_names


# In[12]:


#Transformation des bunch en DataFrame
iris_df = pd.DataFrame(iris.data,columns=iris.feature_names)
wine_df = pd.DataFrame(wine.data,columns=wine.feature_names)


# ### IRIS

# In[13]:


iris_df.head(5)


# In[14]:


print("Dimension du Dataset iris : " , iris_df.shape)
print("\nType de données du Dataset iris :\n", iris_df.dtypes)
print("\nLes caractéristiques  sont :\n", iris_df.columns)
print(" Les Données manquantes pour chaque colonnes :\n", iris_df.isna().sum()) 
#Pas de Valeur manquantes

print('Description statique : ')
iris_df.describe()


# In[15]:


# Description de caractériques
iris_df.boxplot(figsize=(13,5))
plt.title("Distribution des caracteristiques du jeu de données iris")
plt.show()
      


# In[16]:


# Mise à l'echelle StandardScaler
scaler1 = StandardScaler()
iris_standardisee = scaler1.fit_transform(iris_df)
# Réaffichage de la destribution statistque
iris_standardisee_df = pd.DataFrame(iris_standardisee)
iris_standardisee_df.describe()

# Normalisation MinMax
##scaler = MinMaxScaler()
##iris_normalisee = scaler.fit_transform(iris_df)
##iris_normalisee_df = pd.DataFrame(iris_normalisee)
##iris_normalisee_df.describe()


# ### WINE

# In[17]:


wine_df.head()


# In[18]:


# print("Dimension du Dataset wine : " , wine_df.shape)
print("\n\n Type de données du Dataset wine :\n", wine_df.dtypes)
print("\n\nLes Données manquantes pour chaque colonnes :\n", wine_df.isna().sum()) 
#Pas de valeurs manquantes
print("\n\nDescription statistique:\n")
wine_df.describe()


# In[19]:


wine_df.boxplot(figsize=(10,6))
plt.title("Distribution des caracteristiques du jeu de données Wine")
plt.show()
      


# In[20]:


# Mise à l'echelle StandardScaler
wine_standardisee = scaler1.fit_transform(wine_df)
# Réaffichage de la destribution statistque
wine_standardisee_df = pd.DataFrame(wine_standardisee)
wine_standardisee_df.describe()

# Normalisation MinMax


# ### WHOLESALE

# In[21]:


wholesale.head()


# In[22]:


print("Dimension du Dataset wholesale : " , wholesale.shape)
print("\n\nType de données du Dataset wholesale :\n", wholesale.dtypes)
print("\n\nLes Données manquantes pour chaque colonnes :\n", wholesale.isna().sum()) 
#Pas de valeurs manquantes


print("\n\nDescription statistique:\n")
iris_df.describe()


# In[23]:


wholesale.boxplot(figsize=(10,6))
plt.title("Distribution des caracteristiques du jeu de données wholesale")
plt.show()
      


# In[24]:


# Mise à l'echelle StandardScaler
wholesale_standardisee = scaler1.fit_transform(wholesale)
# Réaffichage de la destribution statistque
wholesale_standardisee_df = pd.DataFrame(wholesale_standardisee)
wholesale_standardisee_df.describe()

# Normalisation MinMax


# # PARTIE 2 : Clustering Kmeans 

# ## Kmeans IRIS 

# In[25]:


# Préparation des données
X = iris.data # 
y = iris.target # 


# In[26]:


# Recherche du nombre de clusters appropriés avec la méthode du coude

inerties = []

K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inerties.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, inerties, 'b-',marker="o")
plt.xlabel('nombre de clusters')
plt.ylabel('inerties')
plt.xticks(range(1,len(K)))

plt.show()


# In[28]:


kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)


# In[29]:


print('Centre des points : ', kmeans.cluster_centers_)


# In[30]:


X.shape, y.shape


# In[31]:


labels = kmeans.labels_
print('Labels prédits : \n', labels)


# In[32]:


# Visualisation
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c='red', marker='x')
plt.xlabel('Longueur du sépale')
plt.ylabel('Largeur du sépale')
plt.title('Visualisation des clusters')

plt.show()


# ## Kmeans WINE 

# In[33]:


# Préparation des données
X_wine = wine.data # 
y_wine = wine.target 
# Recherche du nombre de clusters appropriés avec la méthode du coude

inerties = []

K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_wine)
    inerties.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, inerties, 'b-',marker="o")
plt.xlabel('nombre de clusters')
plt.ylabel('inerties')
plt.xticks(range(1,len(K)))

plt.show()


# In[34]:


kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_wine)


# In[35]:


labels = kmeans.labels_
print('Labels prédits : \n', labels)


# In[36]:


# Visualisation
plt.figure(figsize=(8, 6))
plt.scatter(X_wine[:,0], X_wine[:,1], c=labels, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c='red', marker='x')
plt.xlabel('Longueur du sépale')
plt.ylabel('Largeur du sépale')
plt.title('Visualisation des clusters')

plt.show()


# Kmeans Wholesale

# In[39]:


# Préparation des données
X_wholesale = wholesale.iloc[:,:-1] 
y_wholesale = wholesale.iloc[:, -1] 
# Recherche du nombre de clusters appropriés avec la méthode du coude

inerties = []

K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_wholesale)
    inerties.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, inerties, 'b-',marker="o")
plt.xlabel('nombre de clusters')
plt.ylabel('inerties')
plt.xticks(range(1,len(K)))

plt.show()


# In[40]:


kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_wholesale)


# In[41]:


labels = kmeans.labels_
print('Labels prédits : \n', labels)


# In[ ]:





# # PARTIE 4 : DBSCAN

# ### 4.1 IRIS

# In[85]:


# Charger IRIS et le convertir en DataFrame pandas
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

# Création d'un scaler pour standardiser les données
scaler = StandardScaler()


# Standardisation des données IRIS
X_iris_scaled = scaler.fit_transform(df_iris)


# Fonction pour appliquer DBSCAN avec epsilon et min_samples donnés
def apply_dbscan(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)  # Obtenir les étiquettes de cluster
    return labels, dbscan


# Paramètres pour DBSCAN
eps_iris, min_samples_iris = 0.7, 5  # Paramètres pour IRIS

# Appliquer DBSCAN sur les données standardisées
labels_iris, dbscan_iris = apply_dbscan(X_iris_scaled, eps_iris, min_samples_iris)


# Fonction pour tracer les résultats de DBSCAN en utilisant PCA
def plot_dbscan_results(X, labels, title):
    pca = PCA(n_components=2)  # Réduction de dimension à 2D pour visualisation
    X_pca = pca.fit_transform(X)  # Transformation PCA
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Paired')  # Tracer les points colorés par cluster
    plt.title(title)
    plt.show()
    
    
# Visualisation des résultats pour IRIS
plot_dbscan_results(X_iris_scaled, labels_iris, "DBSCAN sur IRIS")



# Fonction pour identifier les outliers (étiquettes -1 sont des outliers)
def get_outliers(labels):
    return np.where(labels == -1)[0]  # Indices des points ayant une étiquette -1

# Identifier et afficher les outliers pour IRIS
outliers_iris = get_outliers(labels_iris)
print(outliers_iris.shape)
print(f"Outliers dans IRIS : {outliers_iris}")



# ### 4.2 WINE

# In[91]:


# Charger WINE et le convertir en DataFrame pandas
wine = load_wine()
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)

# Standardisation des données WINE
X_wine_scaled = scaler.fit_transform(df_wine)

# Paramètres pour DBSCAN WINE
eps_wine, min_samples_wine = 0.5, 10 # Paramètres pour WINE


# Appliquer DBSCAN sur les données standardisées
labels_wine, dbscan_wine = apply_dbscan(X_wine_scaled, eps_wine, min_samples_wine)

# Visualisation des résultats pour WINE 
plot_dbscan_results(X_wine_scaled, labels_wine, "DBSCAN sur WINE")

# Identifier et afficher les outliers pour chaque jeu de données
outliers_wine = get_outliers(labels_wine)
print(outliers_wine.shape)
print(f"Outliers dans WINE : {outliers_wine}")


# ### 4.3 WHOLESALES

# In[92]:


# Standardisation des données Wholesale Customers (en retirant les deux premières colonnes 'Channel' et 'Region')
X_wholesale_scaled = scaler.fit_transform(wholesale.iloc[:, 2:])


# Paramètres pour DBSCAN 
eps_wholesale, min_samples_wholesale = 0.9, 5  # Paramètres pour Wholesale Customers

# Appliquer DBSCAN sur les données standardisées
labels_wholesale, dbscan_wholesale = apply_dbscan(X_wholesale_scaled, eps_wholesale, min_samples_wholesale)


# Visualisation des résultats pour IRIS, WINE et Wholesale Customers
plot_dbscan_results(X_wholesale_scaled, labels_wholesale, "DBSCAN sur Wholesale Customers")


# Identifier et afficher les outliers pour chaque jeu de données
outliers_wholesale = get_outliers(labels_wholesale)
print(outliers_wholesale.shape)
print(f"Outliers dans Wholesale Customers : {outliers_wholesale}")


# In[ ]:




