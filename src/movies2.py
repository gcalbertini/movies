import importlib

from pandas.core.frame import DataFrame
from myModules import movieClass
importlib.reload(movieClass)
import pingouin as pg
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as  plt


Movies = movieClass.movie(verbose=False, alpha=0.005)

usrData = pd.DataFrame(Movies.userData()).T
names = ['User ' + str(i) for i in range(len(usrData.columns))]
usrData.columns = names

corr = usrData.corr().abs()
#keep lower triangular without the diagonal
corr[:] = np.tril(corr.values, k=-1)
corrMax = corr.max()
megaMax = corrMax.max()

pairs = []

i = 0
for usr in names:   
    match = corr.index[corr[usr] == corrMax[i]].tolist()
    pairs.append((usr,*match,'Coefficient of ' + str(corrMax[i])))
    i+=1

corrUsers = []
for i in range(10):
    corrUsers.append(pairs[i][1])
string = "The most correlated users for users 0 to 9 were as follows: {list}.".format(list=corrUsers)
print(string)

for pair in pairs:
    if pair[2] == 'Coefficient of ' + str(megaMax):
        megaPair = (pair[0],pair[1])
        break
string = "The most correlated pair of users is pair {pair} with coefficient of {coeff}.".format(pair=megaPair, coeff=megaMax)
print(string)

#For missing entries, avg from the column used. Personal columns are from 401 to 474. First 400 are movie ratings.
Movies = movieClass.movie(verbose=False,alpha=0.005,fillAvg=True)
data = Movies.columnData(fillAvg=True, dropNan = False)
df_rate = pd.DataFrame(data[:Movies.movieCols])
df_pers = pd.DataFrame(data[Movies.movieCols:-3])

#80% for training set, 20% for test
#Model df_pers = function(df_rate)
x_train,x_test,y_train,y_test=train_test_split(df_rate.T,df_pers.T,test_size=0.2, random_state = 42)

# We're going to focus particularly on sensation-seeking behaviors (Columns 401-421) as they all query from a set of questions that lie on a risky/high energy-to-risk-averse spectrum 
# and will be aggregated to form our dependent vector  

Y_train = y_train.iloc[:,0:20].agg('sum',axis='columns')# dependent variables (really, y_hat + residuals, y = (B_0 * x_0 +...+ B_76 * x_76) + e for ALL users, so multiple OLS)
Y_test = y_test.iloc[:,0:20].agg('sum',axis='columns')

# OLS Model
model = linear_model.LinearRegression().fit(x_train, Y_train)  # fitting the model
#Predict function is model.intercept_ + np.dot(x_train, model.coef_)
yhat_test = model.predict(x_test)
yhat_train = model.predict(x_train)

train_MAE = mean_absolute_error(Y_train.values, yhat_train) 
test_MAE = mean_absolute_error(Y_test.values, yhat_test)

print(train_MAE)
print(test_MAE)

# Ridge Regression Version
alphas = [0, 1e-8, 1e-5, .1, 1, 10]
model_alphas = []
for a in alphas:
    model = linear_model.Ridge(alpha=a).fit(x_train, Y_train)
    yhat_test = model.predict(x_test)
    yhat_train = model.predict(x_train)
    train_MAE = mean_absolute_error(Y_train.values, yhat_train) 
    test_MAE = mean_absolute_error(Y_test.values, yhat_test)
    model_alphas.append((train_MAE,test_MAE))

prime_alpha_ridge = alphas[model_alphas.index(min(model_alphas, key = lambda t: t[1]))]
print(prime_alpha_ridge)

#Lasso Regression Version
alphas = [1e-3,1e-2,1e-1,1]
model_alphas = []
for a in alphas:
    model = linear_model.Lasso(alpha=a, max_iter=10000).fit(x_train, Y_train)
    yhat_test = model.predict(x_test)
    yhat_train = model.predict(x_train)
    train_MAE = mean_absolute_error(Y_train.values, yhat_train) 
    test_MAE = mean_absolute_error(Y_test.values, yhat_test)
    model_alphas.append((train_MAE,test_MAE))

prime_alpha_lasso = alphas[model_alphas.index(min(model_alphas, key = lambda t: t[1]))]
print(prime_alpha_lasso)