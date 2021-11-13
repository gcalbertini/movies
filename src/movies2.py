import importlib
from myModules import movieClass
importlib.reload(movieClass)
import pingouin as pg
import pandas as pd
import numpy as np

Movies = movieClass.movie(verbose=True, alpha=0.005)

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

Movies = movieClass.movie(verbose=False,alpha=0.005,fillAvg=False)
data = Movies.columnData(fillAvg=False, dropNan = False)
print(data)
