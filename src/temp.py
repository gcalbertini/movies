from myModules import movieClass
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
import seaborn as sns
sns.set(style="darkgrid")


Movies = movieClass.movie(verbose=True, alpha=0.005)
usrData = pd.DataFrame(Movies.userData()).T
names = ['User ' + str(i) for i in range(len(usrData.columns))]
usrData.columns = names
data = usrData[420:474].T
factors = Movies.titles[420:474]

# 1. Z-score the data: same as StandardScaler().fit_transform(data.values)
zscoredData = stats.zscore(data)

# 2. Run the PCA:
pca = PCA().fit(zscoredData)

# 3a. Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_

# 3b. Loadings (eigenvectors): Weights per factor in terms of the original data. Where do the
# principal components point, in terms of the 54 questions?
loadings = pca.components_

# 3c. Rotated Data: Simply the transformed data - we had 1097 particpants (rows) in
# terms of 54 variables (each question is a column); now we have 1097 participants in terms of 54
# factors ordered by decreasing eigenvalue
rotatedData = pca.fit_transform(zscoredData)

# 4. For the purposes of this, you can think of eigenvalues in terms of
# (co)variance explained:
covarExplained = eigVals/sum(eigVals)*100
# We note that there about 11/54 factors that explain most of the data in terms of covariance

fig1 = plt.figure()
numClasses = data.shape[1]
plt.bar(np.linspace(1, numClasses, numClasses), eigVals)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
# Kaiser criterion line; 12 factors considered (12th has eigVal 0.98 and is also considered)
plt.plot([0, numClasses], [1, 1], color='red', linewidth=1)
plt.show()

# Using Kaiser criterion: Keep all factors with an eigenvalue > 1
# Rationale: Each variable adds 1 to the sum of eigenvalues. The eigensum.
# We expect each factor to explain at least as much as it adds to what needs
# to be explained. The factors have to carry their weight.
# By this criterion, we would report 10 meaningful factors.

# Now that we realize that 1, 2 or 3 are reasonable solutions to the course
# evaluation issue, we have to interpret the factors.
# This is perhaps where researchers have the most leeway.
# You do this - in principle - by looking at the loadings - in which
# direction does the factor point?

fig2, axs = plt.subplots(6, 2, figsize=(15, 10), sharey=True)
x = np.linspace(1, data.shape[1], data.shape[1])
pc = 0
for row in range(6):
    for col in range(2):
        y = np.abs(loadings[pc, :])
        axs[row, col].bar(x, y)
        title = 'PC' + str(pc+1) + ': ' + \
            str(factors[np.argmax(np.abs(loadings[pc]))])
        axs[row, col].set_title(title[0:95], fontdict={'fontsize': 8}, pad=2.0)
        pc += 1

plt.subplots_adjust(left=0.125,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.05,
                    hspace=0.5)
plt.show()

#Now using the 12 components
pca = PCA(n_components=12).fit(zscoredData)
rotatedData = pca.fit_transform(zscoredData)


#Plots??
#plt.plot(rotatedData[:,0]*-1,rotatedData[:,1]*-1,'o',markersize=5)
#plt.xlabel('Is full of energy')
#plt.ylabel('Emotions rub off on me')
#plt.show()

n_clusters = 25
cost = []
for i in range(1, n_clusters):
    kmean = KMeans(i)
    kmean.fit(rotatedData)
    cost.append(kmean.inertia_)
plt.ylabel('Within Cluster Sum of Squares')
plt.xlabel('Number of Clusters')
plt.title('KMeans with PCA Clustering')
plt.plot(cost, 'bx-')
plt.show()

# see: https://vitalflux.com/kmeans-silhouette-score-explained-with-python-example/
fig3, ax = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
for i in [2, 3, 4, 5]:
    km = KMeans(n_clusters=i, init='k-means++',
                n_init=10, max_iter=400, random_state=42)
    q, mod = divmod(i, 2)
    visualizer = SilhouetteVisualizer(
        km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(rotatedData)
plt.show()

# We select only 2 clusters as we see uniform thickness and few possible misclassifications (due to negative scores) -extra credit (discuss this more)

kmeans_pca = KMeans(n_clusters=2, init='k-means++', random_state=42)
kmeans_pca.fit(rotatedData)
df_segm_pca_kmeans = pd.concat(
    [data.reset_index(drop=False), pd.DataFrame(rotatedData)], axis=1)

df_segm_pca_kmeans.rename(columns={0: 'PC1', 1: 'PC2', 2: 'PC3', 3: 'PC4', 4: 'PC5',
                          5: 'PC6', 6: 'PC7', 7: 'PC8', 8: 'PC9', 9: 'PC10', 10: 'PC11', 11: 'PC12'}, inplace=True)
df_segm_pca_kmeans['Segment KMeans PCA'] = kmeans_pca.labels_
df_segm_pca_kmeans['Segment'] = df_segm_pca_kmeans['Segment KMeans PCA'].map({
                                                                             0: 'first', 1: 'second'})

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
x_axis1 = df_segm_pca_kmeans.loc[df_segm_pca_kmeans['Segment']
                                 == 'first', 'PC1']
y_axis1 = df_segm_pca_kmeans.loc[df_segm_pca_kmeans['Segment']
                                 == 'first', 'PC2']
z_axis1 = df_segm_pca_kmeans.loc[df_segm_pca_kmeans['Segment']
                                 == 'first', 'PC3']
x_axis2 = df_segm_pca_kmeans.loc[df_segm_pca_kmeans['Segment']
                                 == 'second', 'PC1']
y_axis2 = df_segm_pca_kmeans.loc[df_segm_pca_kmeans['Segment']
                                 == 'second', 'PC2']
z_axis2 = df_segm_pca_kmeans.loc[df_segm_pca_kmeans['Segment']
                                 == 'second', 'PC3']
# We will then label the three axes using the percentages explained for each major component.
ax.set_xlabel(
    'PCA-1, ' + str(round(pca.explained_variance_ratio_[0]*100, 2)) + '% Explained', fontsize=9)
ax.set_ylabel(
    'PCA-2, ' + str(round(pca.explained_variance_ratio_[1]*100, 2)) + '% Explained', fontsize=9)
ax.set_zlabel(
    'PCA-3, ' + str(round(pca.explained_variance_ratio_[2]*100, 2)) + '% Explained', fontsize=9)
ax.scatter(x_axis1, y_axis1, z_axis1, marker='x', color='r')
ax.scatter(x_axis2, y_axis2, z_axis2, marker='o', color='b')
plt.legend(labels=['Segment 1', 'Segment 2'])

plt.show()
