from myModules import movieClass
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

Movies = movieClass.movie(verbose=True, alpha=0.005)
usrData = pd.DataFrame(Movies.userData()).T
names = ['User ' + str(i) for i in range(len(usrData.columns))]
usrData.columns = names
data = usrData[420:474].T
factors = Movies.titles[420:474]

# Compute correlation between each measure across all courses:
r = np.corrcoef(data,rowvar=False) # True = variables are rowwise; False = variables are columnwise

# Plot the data:
fig1 = plt.figure()
plt.imshow(r) 
plt.colorbar()
plt.show()
# We see that most of these features (Qs) are low-to-moderately correlated with one another 
# so we expect there to only be a fraction of all 53 components that explain most of the variance of the data

#FROM LAB
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

fig2 = plt.figure()
numClasses = data.shape[1]
plt.bar(np.linspace(1,numClasses,numClasses),eigVals)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,numClasses],[1,1],color='red',linewidth=1) # Kaiser criterion line; 11 factors considered
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

whichPrincipalComponent = 1 # Try a few possibilities (at least 1,2,3 - or 0,1,2 that is - indexing from 0)

# 1: The first one accounts for almost everything, so it will probably point 
# in all directions at once
# 2: Challenging/informative - how much information?
# 3: Organization/clarity: Pointing to 6 and 5, and away from 16 - structure?

plt.bar(np.linspace(1,17,17),loadings[whichPrincipalComponent,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading')

# General principle: Looking at the highest loadings (positive or negative)
# and looking for commonalities.


# For instance, let's say the school wants to figure out which courses are
# good or needlessly hard, we can now look at that

plt.plot(rotatedData[:,0]*-1,rotatedData[:,1]*-1,'o',markersize=5)
plt.xlabel('Overall course quality')
plt.ylabel('Hardness of course')

# In this sense, PCA can help in decision making - are there some classes
# that are under/over-performing, given their characteristics?
# If we had more than 40 courses, looking at the 3rd dimension would be
# interesting too. As is, it is a bit sparse.
