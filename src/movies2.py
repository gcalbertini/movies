from myModules import movieClass
import pingouin as pg
import pandas as pd

Movies = movieClass.movie(verbose=True, alpha=0.005)

usrData = pd.DataFrame(Movies.userData()).T
cols = ['User ' + str(i+1) for i in range(len(usrData.columns))]
usrData.columns = cols

corr = pg.rcorr(usrData, method='pearson')
pairs = {}

    

