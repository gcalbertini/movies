from os import error
import pandas as pd
import statistics as st
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import random
import re
import itertools


#Movie class -- future work: need to fix 'error(...)' lines to break execution

class movie:

    def __init__(self, dataset="data/movieReplicationSet.csv", alpha = 0.05, verbose = True, movieCols = 400, fillAvg = True):
        self.alpha = alpha
        self.movieCols = movieCols

        try: 
            self.dataset = pd.read_csv(dataset)
        except FileNotFoundError:
            error("File not found!")

        self.movies= dict(itertools.islice(self.table(fillAvg).items(), movieCols)) 
        self.verbose = True
        self.titles = list(self.dataset.columns)

    def franchiseFinder(self,str):
        table = self.table(dropNan = True, moviesOnly = True)
        self.franchises = {}
        for title, ratings in table.items():
            if str in title:
                self.franchises[title]=ratings 
        return self.franchises

    def franchiseDiff(self, franchisesList, hyp = 'two-sided', text = "franchise shows inconsitent quality across its movie ratings.", alpha_ratio = 1):
        if franchisesList == None: error("At least one franchise must be entered!")
        self.verbose = False
        self.diff = [] #keep a list of discrepant distributions using KS test
        for franchiseName in franchisesList:
            #franchise iterating over list of franchise input values
            franchise = self.franchiseFinder(franchiseName) #now becomes all key-value pairs of movies for that franchise [A1:1,A2:2,A3:3,A4:4]
            #now a list of all values (ratings data) for those franchise films [1,2,3,4]
            data = list(franchise.values())
            count = len(data) #4 in this example
            start_compared = 0
            next_compared = start_compared+1
            res = franchiseName + ' franchise does not show inconsistent ' + text
            while start_compared < count-1:
            #compare data sets 1-2, 1-3, 1-4 | 2-3, 2-4 | 3-4
                
                pval = self.kstest2(data[start_compared], data[next_compared], hyp)
                next_compared+=1

                if pval[1] < alpha_ratio*self.alpha:
                    res = franchiseName + ' franchise shows INCONSISTENT ' + text
                    break 

                if next_compared == count:
                    start_compared+=1
                    next_compared = start_compared+1

            self.diff+=[res] 

        self.verbose = True
        return self.diff


    #Note: "dropNan" will drop non-numeric values from the number values associated with each column from the spreadsheet,
    #thus will not do row-wise element elimination for blanks or NAN"
    def columnData(self, dropNan = False, fillAvg = False):
        self.df = self.dataset.values
        self.data = []
        for col in range(self.df.shape[1]):
            vec = []
            for row in range(self.df.shape[0]):
                vec.append(self.df[row,col])
            self.data.append(vec)
        self.data = np.array(self.data) 

        if dropNan == True and fillAvg ==True: error('Cannot both drop NAN and fill NAN values with column averages. Check default parameter settings.')

        if dropNan == True:
            self.data_dropnan = []
            for entry in self.data: self.data_dropnan.append(entry[~np.isnan(entry)])
            return self.data_dropnan
        elif dropNan == False and fillAvg == True:
            self.data_avg = []
            for entry in self.data: self.data_avg.append(np.nan_to_num(entry, nan=np.nanmean(entry), copy=False))
            return self.data_avg
        elif dropNan == False and fillAvg == False:
            return self.data


    def userData(self, moviesOnly = False):
        # User data (row data) will fill in NAN entries with averages of the columns
        self.df = self.columnData(fillAvg = True)
        self.data = []

        if not moviesOnly == True:
            condition = len(self.df)
        else:
            condition = self.movieCols

        for user in range(self.dataset.shape[0]):
            userVec = []
            for col in range(condition):
                userVec.append(self.df[col][user])
            self.data.append(userVec)

        return np.array(self.data) 

    def table(self, dropNan = False, fillAvg = False, moviesOnly = False):
        self.dict = {}
        if dropNan == True and fillAvg == True: error('Cannot both drop NAN and fill NAN values with column averages. Check default parameter settings.')

        if dropNan == True:
            data = self.columnData(dropNan == True)
        elif dropNan == False and fillAvg == False:
            data = self.columnData()
        elif dropNan == False and fillAvg == True: 
            data = self.columnData(fillAvg == True)
        
        self.titles = list(self.dataset.columns)
        d = 0
        if moviesOnly == False:
            for title in self.titles:
                self.dict.__setitem__(title, data[d]) 
                d+=1
            return self.dict
        else:
            for title in self.titles[:self.movieCols]:
                self.dict.__setitem__(title, data[d]) 
                d+=1
            return self.dict

    def popularity(self, colTitle = "All"):
        self.entry = colTitle
        titles = self.titles
        titles = titles[:self.movieCols]
        validMovie = [colTitle in titles or colTitle == "All"]
        self.popularities = []
        if validMovie:
            if colTitle == "All":
                for reviews in self.movies.values():
                    self.popularities.append(len(reviews))
            else:
                for key, reviews in self.movies.items():
                    if key == colTitle:
                        self.popularities.append(len(reviews))
        else:
            error("Title of film not in column headers.")

        return self.popularities
    
    def plot(self,x,y, name = "FIGUREX", n_bins = 10,titleX="Xtitle",titleY="Ytitle",x1="xlbl1",x2="xlbl2",y1="ylbl1",y2="ylbl2"):
        self.n_bins = n_bins
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        axs[0].set_title(titleX)
        axs[1].set_title(titleY)
        axs[0].set_ylabel(x1)
        axs[1].set_ylabel(y1)
        axs[0].set_xlabel(x2)
        axs[1].set_xlabel(y2)
        axs[0].hist(x, bins=n_bins)
        axs[1].hist(y, bins=n_bins)
        fig.tight_layout()
        fig.savefig(name,dpi=200) 
        plt.show()

    #Two sample t-test (unpaired or independent t-test)
    #The two-sample (unpaired or independent) t-test compares the means of two independent groups,
    #determining whether they are equal or significantly different. In two sample t-test, usually,
    #we compute the sample means from two groups and derives the conclusion for the population’s
    #means (unknown means) from which two groups are drawn.
    def ttest2(self, x,y,hyp = 'two-sided', text = "that <what you try to test>"):
        var1 = st.variance(x)
        var2 = st.variance(y)
        varianceEqual = [abs((var1-var2)/var1) < 0.1]
        self.val = stats.ttest_ind(x, y, axis=0, equal_var=varianceEqual, nan_policy='raise', alternative=hyp)
        if self.verbose:
            if self.val[1] < self.alpha:
                s = "sufficient"
                eq = "<"
                response = "reject"
            else:
                s = "insufficient"
                eq = ">"
                response = "fail to reject"

            print("As p-value of {pval} is {sign} alpha of {alpha} at test statistic {stat}, we {res} the null hypothesis.\nThere is {suf} evidence to suggest that {analysis}\n".format(sign = eq, \
                pval = format(self.val[1],".5f"), stat = format(self.val[0],".2f"), alpha= self.alpha, suf = s, res = response, analysis=text.lstrip()))     
        else:
            return self.val

    #Assumptions for U test: You should have independence of observations, which means that there is no relationship between 
    # the observations in each group of the independent variable or between the groups themselves. For example, there must be different participants 
    # in each group with no participant being in more than one group. 
    # The distribution of scores for (example) "males" and the distribution of scores for "females" for the 
    # independent variable, "gender") MUST have roughly the same shape.
   
    def utest2(self, x,y, hyp = 'two-sided', text = "that <what you try to test>"):
        self.val = stats.mannwhitneyu(x, y, alternative=hyp, method='auto')
        if self.verbose:
            if self.val[1] < self.alpha:
                s = "sufficient"
                eq = "<"
                response = "reject"
            else:
                s = "insufficient"
                eq = ">"
                response = "fail to reject"

            print("As p-value of {pval} is {sign} alpha of {alpha} at test statistic {stat}, we {res} the null hypothesis.\nThere is {suf} evidence to suggest that {analysis}\n".format(sign = eq, \
                pval = format(self.val[1],".5f"), stat = format(self.val[0],".2f"), alpha= self.alpha, suf = s, res = response, analysis=text.lstrip()))     
        else:
            return self.val


    def kstest2(self, x,y, hyp = 'two-sided', text = "that <what you try to test>"):
        self.val = stats.ks_2samp(x, y, alternative=hyp, mode='auto')
        if self.verbose:
            if self.val[1] < self.alpha:
                s = "sufficient"
                eq = "<"
                response = "reject"
            else:
                s = "insufficient"
                eq = ">"
                response = "fail to reject"

            print("As p-value of {pval} is {sign} alpha of {alpha} at test statistic {stat}, we {res} the null hypothesis.\nThere is {suf} evidence to suggest that {analysis}\n".format(sign = eq, \
                pval = format(self.val[1],".5f"), stat = format(self.val[0],".2f"), alpha= self.alpha, suf = s, res = response, analysis=text.lstrip()))     
        else:
            return self.val

    def movieYrs(self):
        titles = self.titles
        titles = titles[:self.movieCols]
        self.years = []
        for title, data in self.movies.items():
            self.years.append(re.findall(r'\d+', title)) #fix [['2001'],['1995'],...] if you have time
        self.years = [i[0] for i in self.years]
        return list(map(int, self.years))
    
    #eliminate row-wise data wrt to two data columns
    def rowElim(self, col_title_1, col_title_2):
        df = self.dataset #Movies.dataset would init another class to return the same member in the class
        mod_df = df.dropna( how='any', subset=[col_title_1, col_title_2])
        return np.array(mod_df[col_title_1]), np.array(mod_df[col_title_2])

    def utest2_prop(self, xcol_title = 'Gender identity (1 = female; 2 = male; 3 = self-described)', options = [1, 2, 3], hyp = 'two-sided', text="that <what you try to test>", verbose=False):
        #TO-DO: Change this for a certain test (not only u-test)
        count = 0
        df = self.dataset
        self.pvals = []
        titles = list(df.columns)
        self.movies_clean = []
        for title in titles[:self.movieCols]:
            ref , y = self.rowElim(xcol_title, title)
            self.movies_clean.append((ref,y))
        
        self.sig_diff=0
        for data_pairs in self.movies_clean:
            self.A_rating = []
            self.B_rating = []
            ref = data_pairs[0] #gender, single status, etc
            ratings = data_pairs[1]
            self.verbose = verbose

            for i in range(len(ref)):
                count = 0
                if ref[i] == options[0]:
                    self.A_rating.append(ratings[i])
                    count+=1
                elif ref[i] == options[1]:
                    self.B_rating.append(ratings[i])
                    count+=1
                #due to scarcity of additional info on self-described individuals, randomly assigned for them
                elif ref[i] == options[2]:
                    count+=1
                    choice = random.randint(0, 1)
                    if choice:
                        self.A_rating.append(ratings[i])
                    else:
                        self.B_rating.append(ratings[i])
                else: error('DATA MISMATCH')     

            if count!=len(self.A_rating)+len(self.B_rating):error('POSSIBLE DATA MISMATCH')

            val = self.utest2(self.A_rating,self.B_rating, hyp, text)
            self.pvals.append(val[1])

            if val[1] < self.alpha:
                self.sig_diff+=1

        self.prop = self.sig_diff/self.movieCols
               
        print("About {p}% of movies ({count}) {analysis}\n".format(p = format(100*self.prop,".2f"), analysis = text, count = int(self.prop*self.movieCols)))
        self.verbose = True #revert to default
        return self.prop, self.pvals
