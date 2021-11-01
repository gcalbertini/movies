from os import error
import pandas as pd
import statistics as st
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import random
import re
import itertools


#Movie class -- future work: introduce sets (dict are unordered but structure here v ordered), debug, more clever algos

class movie:

    def __init__(self, dataset="movieReplicationSet.csv", alpha = 0.05, verbose = True, movieCols = 400):
        self.alpha = alpha
        self.movieCols = movieCols

        try: 
            self.dataset = pd.read_csv(dataset)
        except FileNotFoundError:
            error("File not found!")

        self.movies= dict(itertools.islice(self.table(dropNan=True).items(), movieCols)) 
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
    def columnData(self, dropNan = False):
        self.df = self.dataset.values
        self.data = []
        for col in range(self.df.shape[1]):
            vec = []
            for row in range(self.df.shape[0]):
                vec.append(self.df[row,col])
            self.data.append(vec)
        self.data = np.array(self.data) 

        if dropNan == False:
            return self.data
        else: 
            self.data_dropnan = []
            for entry in self.data: self.data_dropnan.append(entry[~np.isnan(entry)])
            return self.data_dropnan

    def table(self, dropNan = False, moviesOnly = False):
        self.dict = {}
        data = self.columnData(dropNan)
        self.titles = list(self.dataset.columns)
        d = 0
        if not moviesOnly:
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
                res = "reject"
            else:
                s = "insufficient"
                eq = ">"
                res = "fail to reject"

            print("As p-value of {pval} is {sign} alpha of {alpha} at test statistic {stat}, we {res} the null hypothesis.\nThere is {suf} evidence to suggest that {analysis}\n".format(sign = eq, \
                pval = format(self.val[1],".5f"), stat = format(self.val[0],".2f"), alpha= self.alpha, suf = s, res = res, analysis=text.lstrip()))     
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
                res = "reject"
            else:
                s = "insufficient"
                eq = ">"
                res = "fail to"

            print("As p-value of {pval} is {sign} alpha of {alpha} at test statistic {stat}, we {res} the null hypothesis.\nThere is {suf} evidence to suggest that {analysis}\n".format(sign = eq, \
                pval = format(self.val[1],".5f"), stat = format(self.val[0],".2f"), alpha= self.alpha, suf = s, res = res, analysis=text.lstrip()))     
        else:
            return self.val


    def kstest2(self, x,y, hyp = 'two-sided', text = "that <what you try to test>"):
        self.val = stats.ks_2samp(x, y, alternative=hyp, mode='auto')
        if self.verbose:
            if self.val[1] < self.alpha:
                s = "sufficient"
                eq = "<"
                res = "reject"
            else:
                s = "insufficient"
                eq = ">"
                res = "fail to reject"

            print("As p-value of {pval} is {sign} alpha of {alpha} at test statistic {stat}, we {res} the null hypothesis.\nThere is {suf} evidence to suggest that {analysis}\n".format(sign = eq, \
                pval = format(self.val[1],".5f"), stat = format(self.val[0],".2f"), alpha= self.alpha, suf = s, res = res, analysis=text.lstrip()))     
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

       
            
#=========================================================================================================================
#=========================================================================================================================

Movies = movie()
#Some useful dictionaries
movies_clean = Movies.table(dropNan=True, moviesOnly=True)
movies_col = Movies.columnData(dropNan=True)

#=========================================================================================================================
#=========================================================================================================================

# 1. Are movies that are more popular (operationalized as having more ratings) rated higher than movies that are less popular?
movies_popularities = Movies.popularity()
median_pop = st.median(movies_popularities)
populars = []
sleepers = []

#median-split of movie popularities
for i in range(len(movies_popularities)):
    if movies_popularities[i] > median_pop:
        populars.append(movies_col[i])
    elif movies_popularities[i] < median_pop:
        sleepers.append(movies_col[i])
    if movies_popularities[i] == median_pop:
        choice = random.randint(0, 1)
        if choice:
            populars.append(movies_col[i])
        else:
            sleepers.append(movies_col[i])
    
if (len(sleepers)+len(populars)!=400):error("DATA MISMATCH")
            
sample_means_populars = []
sample_means_sleepers = []
for movie in populars: sample_means_populars.append(np.mean(movie))
for movie in sleepers: sample_means_sleepers.append(np.mean(movie))

Movies.plot(sample_means_populars, sample_means_sleepers, "Q1", n_bins = 10, titleX = "Popular Movie Averages",titleY = "Unpopular Movie Averages", \
    x1="Reviews",y1="Reviews",x2="Avg Rating",y2="Avg Rating")

pval1 = Movies.ttest2(sample_means_populars,sample_means_sleepers, hyp = 'greater', text = "that movies that are more \
popular have ratings that are higher than movies that are less popular.")

#=========================================================================================================================
#=========================================================================================================================

# 2. Are movies that are newer rated differently than movies that are older? 
movie_yrs = Movies.movieYrs()
mean_yrs = np.mean(movie_yrs)
sample_means_movies = []
for ratings in movies_col: sample_means_movies.append(np.mean(ratings))

#median-split of movie years
newer = []
older = []
median_age = st.median(movie_yrs)
for i in range(len(movie_yrs)):
    if movie_yrs[i] > median_age:
        newer.append(sample_means_movies[i])
    elif movie_yrs[i] < median_age:
        older.append(sample_means_movies[i])

    if movie_yrs[i] == median_age:
        choice = random.randint(0, 1)
        if choice:
            newer.append(sample_means_movies[i])
        else:
            older.append(sample_means_movies[i])

if (len(newer)+len(older)!=400):error("DATA MISMATCH")

Movies.plot(older, newer, "Q2", n_bins = 12, titleX = "Older Movie Averages",titleY = "Newer Movie Averages", \
    x1="Counts",y1="Counts",x2="Avg Rating",y2="Avg Rating")
   
pval2 = Movies.utest2(newer,older, hyp = 'two-sided', text = "that newer movies are rated differently than older films.")  

#=========================================================================================================================
#=========================================================================================================================

#3. Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently? 
shrek_males = [] 
shrek_females = []
#row-wise elimination of entries needed as gender may have NAN for entries movie does not and vice-versa
genders, ratings_shrek = Movies.rowElim('Gender identity (1 = female; 2 = male; 3 = self-described)','Shrek (2001)')

count = 0
for i in range(len(genders)):
   
    if genders[i] == 1:
        shrek_females.append(ratings_shrek[i])
        count+=1
    elif genders[i] == 2:
        shrek_males.append(ratings_shrek[i])
        count+=1
    #due to scarcity of additional info on self-described individuals, gender randomly assigned for them
    elif genders[i] == 3:
        count+=1
        choice = random.randint(0,1)
        if choice:
            shrek_males.append(ratings_shrek[i])
        else:
            shrek_females.append(ratings_shrek[i])
    else: error('GENDER MISMATCH')
if count!=len(shrek_females)+len(shrek_males):error('POSSIBLE DATA MISMATCH')

Movies.plot(shrek_males, shrek_females, "Q3", n_bins = 6, titleX = "Male Ratings",titleY = "Female Ratings", \
    x1="Counts",y1="Counts",x2="Rating",y2="Rating")

pval3 = Movies.utest2(shrek_males,shrek_females, hyp = 'two-sided', text = "that enjoyment of Shrek (2001) is gendered.")  
#=========================================================================================================================
#=========================================================================================================================

# 4. What proportion of movies are rated differently by male and female viewers? 
prop4, pvals4 = Movies.utest2_prop(xcol_title = 'Gender identity (1 = female; 2 = male; 3 = self-described)', \
    options = [1, 2, 3], hyp = 'two-sided', text="show gendered preferences.")

#=========================================================================================================================
#=========================================================================================================================

#5. Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings? 
LK_single = [] 
LK_multi = []
#row-wise elimination of entries needed as gender may have NAN for entries movie does not and vice-versa
status, ratings_LK = Movies.rowElim('Are you an only child? (1: Yes; 0: No; -1: Did not respond)','The Lion King (1994)')

count = 0
for i in range(len(status)):
    if status[i] == 1:
        LK_single.append(ratings_LK[i])
        count+=1
    elif status[i] == 0:
        LK_multi.append(ratings_LK[i])
        count+=1
    #neglect the few who did not respond as there is not a guaranteed "50/50" chance someone has siblings 
if count!=len(LK_single)+len(LK_multi):error('POSSIBLE DATA MISMATCH')

Movies.plot(LK_single, LK_multi, "Q4", n_bins = 8, titleX = "Lion King (1994) with Single Children",titleY = "Lion King (1994) with Sibilings", \
    x1="Counts",y1="Counts",x2="Rating",y2="Rating")

pval5 = Movies.utest2(LK_single, LK_multi, hyp = 'greater', text = "single children without siblings enjoy Lion King (1994) more.")  
#=========================================================================================================================
#=========================================================================================================================

#6. What proportion of movies exhibit an “only child effect”, i.e. are rated different by viewers with siblings  vs. those without?  
pval6, pvals6 = Movies.utest2_prop(xcol_title = 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)', \
    options = [1, 0, 999], hyp = 'two-sided', text="are rated differently by viewers with siblings vs. those without.")
    #Note: 999 signals to neglect the ones who did responded as we don't have a 50/50 chance one has siblings or not

#=========================================================================================================================
#=========================================================================================================================

#7. Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who prefer to watch them alone? 
WW_alone = [] 
WW_ppl = []
#row-wise elimination of entries needed as gender may have NAN for entries movie does not and vice-versa
status, ratings_WW = Movies.rowElim('Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)','The Wolf of Wall Street (2013)')

count = 0
for i in range(len(status)):
    if status[i] == 1:
        WW_alone.append(ratings_WW[i])
        count+=1
    elif status[i] == 0:
        WW_ppl.append(ratings_WW[i])
        count+=1
if count!=len(WW_alone)+len(WW_ppl):error('POSSIBLE DATA MISMATCH')
pval7= Movies.utest2(WW_alone, WW_ppl, hyp = 'less', text = "people enjoy watching the The Wolf of Wall Street (2013) with others than alone.")  

Movies.plot(WW_alone, WW_ppl, "Q7", n_bins = 8, titleX = "W.W.S. (2013) - Viewed Alone",titleY = "W.W.S. (2013) - Viewed Together", \
    x1="Counts",y1="Counts",x2="Rating",y2="Rating")
#=========================================================================================================================
#=========================================================================================================================
#8. What proportion of movies exhibit such a “social watching” effect? 
prop8, pvals8 = Movies.utest2_prop(xcol_title = 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)', \
    options = [1, 0, 999], hyp = 'two-sided', text="show a social watching effect.")
#=========================================================================================================================
#=========================================================================================================================
#9. Is the ratings distribution of ‘Home Alone (1990)’ different than that of ‘Finding Nemo (2003)’?  
ratings_FN = [] 
ratings_HA = []

#row-wise elimination of entries needed as gender may have NAN for entries movie does not and vice-versa
ratings_FN, ratings_HA = Movies.rowElim('Finding Nemo (2003)','Home Alone (1990)')

Movies.plot(WW_alone, WW_ppl, "Q9", n_bins = 7, titleX = "Finding Nemo (2003)",titleY = "Home Alone (1990)", \
    x1="Counts",y1="Counts",x2="Rating",y2="Rating")

pval9 = Movies.kstest2(ratings_HA,ratings_FN,text="that the two distributions are different.")
#=========================================================================================================================
#=========================================================================================================================

# There are ratings on movies from several franchises ([‘Star Wars’, ‘Harry Potter’, ‘The Matrix’, ‘Indiana  Jones’, ‘Jurassic Park’,  ‘Pirates of the Caribbean’, ‘Toy Story’, ‘Batman’]) in this dataset. 
# How many of these are of inconsistent quality, as experienced by viewers? [Hint: You can use the keywords in quotation marks 
# featured in this question to identify the movies that are part of each franchise]  

franchisesList = ['Star Wars', 'Harry Potter', 'The Matrix', 'Indiana Jones', 'Jurassic Park', 'Pirates of the Caribbean', 'Toy Story', 'Batman']
discrepant10 = Movies.franchiseDiff(franchisesList, hyp = 'two-sided', text = "quality across its movie ratings.", alpha_ratio= 1)
[print(res) for res in discrepant10]

#=========================================================================================================================
#=========================================================================================================================
#Bonus: Tell us something interesting and true (supported by a significance test of some kind) about the 
#movies in this dataset that is not already covered by the questions above [for 5% of the grade score].

#anderson 2test to see how many of the movie ratings follow normal dist at 5%
#anderson 1test to see how many movie ratings for males/fem follow exponential dist vs normal dist (repeat 1 sample twice)

#TO DO - remove class from implementation + jupyter transfer