import importlib
from myModules import movieClass
importlib.reload(movieClass)
import pingouin as pg
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


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

#For missing entries, avg from the column used. Personal columns are from 401 to 474.
Movies = movieClass.movie(verbose=False,alpha=0.005,fillAvg=True)
data = Movies.columnData(fillAvg=True, dropNan = False)
df_rate = pd.DataFrame(data[:Movies.movieCols])
df_pers = pd.DataFrame(data[Movies.movieCols:-3])

#80% for training set, 20% for test
#df_pers = function(df_rate)
x_train,x_test,y_train,y_test=train_test_split(df_rate.T,df_pers.T,test_size=0.2, random_state = 42)

Y = y_train # dependent variables (really, y_hat + residuals, y = (B_0 * x_0 +...+ B_76 * x_76) + e for ALL users, so multiple yhats)
X = x_train # independent variables (predictors, "betas")
model = linear_model.LinearRegression().fit(X, Y) # fitting the model
yhat = model.predict(x_test)
yhat_odd = model.predict(X)


def loss(pd_Y, np_Yhat):
    y = pd_Y.values
    diff = np.absolute(y-np_Yhat)
    err = []
    for d in diff:
        e = np.average(d)
        err.append(e)
    return np.average(err)


error_train_avg = loss(Y,yhat_odd)
error_test_avg = loss(y_test,yhat)
