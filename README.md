## MOVIES - Project 1/3: Hypothesis Testing

### Purpose: In this project, I will demonstrate the essential skills involved in hypothesis testing. To do so, I
- Will use a real dataset that stems from a replication attempt of published research (Wallisch & Whritner, 2017)
- Set the per-test significance level 𝛼 to 0.005 (as per Benjamin et al., 2018)
- Answer the following:
  1) Are movies that are more popular (operationalized as having more ratings) rated higher than movies that
are less popular?
  2) Are movies that are newer rated differently than movies that are older?
  3) Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently?
  4) What proportion of movies are rated differently by male and female viewers?
  5) Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings?
  6) What proportion of movies exhibit an “only child effect”, i.e. are rated different by viewers with siblings
vs. those without?
  7) Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who
prefer to watch them alone?
  8) What proportion of movies exhibit such a “social watching” effect?
  9) Is the ratings distribution of ‘Home Alone (1990)’ different than that of ‘Finding Nemo (2003)’?
  10) There are ratings on movies from several franchises ([‘Star Wars’, ‘Harry Potter’, ‘The Matrix’, ‘Indiana
Jones’, ‘Jurassic Park’, ‘Pirates of the Caribbean’, ‘Toy Story’, ‘Batman’]) in this dataset. How many of these
are of inconsistent quality, as experienced by viewers? [Hint: You can use the keywords in quotation marks
featured in this question to identify the movies that are part of each franchise]

### Results are found __[here](https://github.com/gcalbertini/movies/blob/9bdc85d0b7215a8d74749e07631c31de902f0023/project1.ipynb)__. The data can be found __[here](https://github.com/gcalbertini/movies/blob/9bdc85d0b7215a8d74749e07631c31de902f0023/data/movieReplicationSet.csv)__.
---

## MOVIES - Project 2/3: Correlation and Regression

### Purpose: In this project, I study correlation and regression in this dataset. To do so, I
- Will use a real dataset that stems from a replication attempt of published research (Wallisch & Whritner, 2017)
- Set the per-test significance level 𝛼 to 0.005 (as per Benjamin et al., 2018)
- Answer the following:
  1) For every user in the given data, find its most correlated user. 
  2) What is the pair of the most correlated users in the data? 
  3) What is the value of this highest correlation?
  4) For users 0, 1, 2, \..., 9, print their most correlated users. 

We want to find a model between the ratings and the personal part of the data. To do so, consider:
- Columns 1-400: These columns contain the ratings for the 400 movies (0 to 4, and missing); call this part `df_rate` and the part of the data which includes all users over columns 401-474
- Columns 401-421: These columns contain self-assessments on sensation seeking behaviors (1-5)
- Columns 422-464: These columns contain responses to personality questions (1-5)
- Columns 465-474: These columns contain self-reported movie experience ratings (1-5)
call this part `df_pers`.


Our main task is to model: 


`df_pers = function(df_rate)`


**Note:** Split the original data into training and testing as the ratio 0.80: 0.20. 

2.1. Model `df_pers = function(df_rate)` by using the linear regression. What are the errors on: (i) the training part; (ii) the testing part?


2.2. Model `df_pers = function(df_rate)` by using the ridge regression with hyperparamter values alpha from [0.0, 1e-8, 1e-5, 0.1, 1, 10]. 

For every of the previous values for alpha, what are the errors on: (i) the training part; (ii) the testing part?

What is a best choice for alpha?


2.3. Model `df_pers = function(df_rate)` by using the lasso regression with hyperparamter values alpha from [1e-3, 1e-2, 1e-1, 1]. 

For every of the previous values for alpha, what are the errors on: (i) the training part; (ii) the testing part?

What is a best choice for alpha?

### Results are found __[here](https://github.com/gcalbertini/movies/blob/9bdc85d0b7215a8d74749e07631c31de902f0023/project2.ipynb)__. The data can be found __[here](https://github.com/gcalbertini/movies/blob/9bdc85d0b7215a8d74749e07631c31de902f0023/data/movieReplicationSet.csv)__.

---

## MOVIES - Project 3/3: ML Methods

### Purpose: In this project, I will demonstrate machine learning methods as applied to the data. We revisit the same dataset from before.

3.1. Apply dimension reduction methods – specifically a PCA – to the data in columns 421-474. As laid out
above, these columns contain self-report answers to personality and how these individuals
experience movies, respectively. 
  - Determine the number of factors (principal components) that we will interpret meaningfully (by
the Kaiser criterion).
  - Semantically interpret what those factors represent. Explicitly name the factors found.
3.2. Plot the data from columns 421-474 in the new coordinate system, where each dot represents a
person, and the axes represent the factors you found in 3.1. 
3.3. Identify clusters in this new space. Use an ML method (e.g. kMeans, DBScan, hierarchical
clustering) to do so. Determine the optimal number of clusters and identify which cluster a given user
is part of.
3.4. Use these principal components and/or clusters identified to build a classification model
(e.g. logistic regression, kNN, SVM, random forest), where we predict the movie ratings of all
movies from the personality factors identified before. Make sure to use cross-validation methods to
avoid overfitting and assess the accuracy of your model by stating its AUC.
3.5. Create a neural network model of your choice to predict movie ratings, using information from all 477
columns. Make sure to comment on the accuracy of this model.

### Results are found __[here](https://github.com/gcalbertini/movies/blob/9bdc85d0b7215a8d74749e07631c31de902f0023/project3.ipynb)__. The data can be found __[here](https://github.com/gcalbertini/movies/blob/9bdc85d0b7215a8d74749e07631c31de902f0023/data/movieReplicationSet.csv)__.

---


#### Dataset description: This dataset features ratings data of 400 movies from 1097 research participants.
- 1st row: Headers (Movie titles/questions) – note that the indexing in this list is from 1
- Row 2-1098: Responses from individual participants
- Columns 1-400: These columns contain the ratings for the 400 movies (0 to 4, and missing)
- Columns 401-421: These columns contain self-assessments on sensation seeking behaviors (1-5)
- Columns 422-464: These columns contain responses to personality questions (1-5)
- Columns 465-474: These columns contain self-reported movie experience ratings (1-5)
- Column 475: Gender identity (1 = female, 2 = male, 3 = self-described)
- Column 476: Only child (1 = yes, 0 = no, -1 = no response)
- Column 477: Movies are best enjoyed alone (1 = yes, 0 = no, -1 = no response)
