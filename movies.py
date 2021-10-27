from os import error
import pandas as pd
import statistics as st
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import random
import re
import itertools


#Movie class -- future work: introduce sets (dict are unordered but structure here v ordered), debug, more clever algos, use pandas vs numpy everything

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
        titles = list(self.dataset.columns)
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
        n_bins = 10
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
                res = "reject"
                eq = "<"
            else:
                s = "insufficient"
                res = "fail to reject"
                eq = ">"

            print("As p-value of {pval} is {sign} alpha of {alpha} at test statistic {stat}, we {res} the null hypothesis.\nThere is {suf} evidence to suggest that {analysis}".format(sign = eq, \
                pval = format(self.val[1],".5f"), stat = format(self.val[0],".3f"), alpha= self.alpha, suf = s, res = res, analysis=text.lstrip()))     
        else:
            return self.val
   
    def utest2(self, x,y, hyp = 'two-sided', text = "that <what you try to test>"):
        self.val = stats.mannwhitneyu(x, y, alternative='two-sided', method='auto')
        if self.verbose:
            if self.val[1] < self.alpha:
                s = "sufficient"
                res = "reject"
                eq = "<"
            else:
                s = "insufficient"
                res = "fail to reject"
                eq = ">"

            print("As p-value of {pval} is {sign} alpha of {alpha} at test statistic {stat}, we {res} the null hypothesis.\nThere is {suf} evidence to suggest that {analysis}".format(sign = eq, \
                pval = format(self.val[1],".5f"), stat = format(self.val[0],".3f"), alpha= self.alpha, suf = s, res = res, analysis=text.lstrip()))     
        else:
            return self.val

    def movieYrs(self):
        titles = list(self.dataset.columns)
        titles = titles[:self.movieCols]
        self.years = []
        for title, data in self.movies.items():
            self.years.append(re.findall(r'\d+', title)) #fix [['2001'],['1995'],...] if you have time
        self.years = [i[0] for i in self.years]
        return list(map(int, self.years))
    
    def rowElimWRT_ref_col(self, ref_col_title, compared_col_title):
        df = Movies.dataset
        mod_df = df.dropna( how='any', subset=[ref_col_title, compared_col_title])
        return np.array(mod_df[ref_col_title]), np.array(mod_df[compared_col_title])
   

Movies = movie()
#Some useful dictionaries
data_clean = Movies.table(dropNan=True)
movies_clean = Movies.table(dropNan=True, moviesOnly=True)

# read csv file
# input = pd.read_csv("movieReplicationSet.csv")
# df = input.to_numpy()
# data = []
# for col in range(df.shape[1]):
#     vec = []
#     for row in range(df.shape[0]):
#         vec.append(df[row,col])
#     data.append(vec)
# data = np.array(data)

#movies_clean = []
#for entry in data: movies_clean.append(entry[~np.isnan(entry)])

movies_popularities = Movies.popularity()
median_pop = st.median(movies_popularities)
populars = []
sleepers = []
movies_col = Movies.columnData(dropNan=True)
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

Movies.plot(sample_means_populars, sample_means_sleepers, "Q1", n_bins = 10, titleX = "sample_means_populars",titleY = "sample_means_sleepers", \
    x1="Counts",y1="Counts",x2="Avg Rating",y2="Avg Rating")

#pval_q1 = stats.ttest_ind(sample_means_populars, sample_means_sleepers, axis=0, equal_var=False, \
#nan_policy='raise', permutations=None, random_state=None, alternative='greater', trim=0)
#print(pval_q1)
pval1 = Movies.ttest2(sample_means_populars,sample_means_sleepers, hyp = 'greater', text = "that movies that are more \
popular have ratings that are higher than movies that are less popular.")

# Are movies that are newer rated differently than movies that are older? 
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

Movies.plot(older, newer, "Q2", n_bins = 10, titleX = "sample_means_older",titleY = "sample_means_newer", \
    x1="Counts",y1="Counts",x2="Avg Rating",y2="Avg Rating")

#pval_q2 = stats.mannwhitneyu(newer, older, alternative='two-sided', method='auto')
#print(pval_q2)     
pval2 = Movies.utest2(newer,older, hyp = 'two-sided', text = "that newer movies are rated differently than older films.")  

# Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently? 

shrek_males = [] 
shrek_females = []
#row-wise elimination of entries needed as gender may have NAN for entries movie does not and vice-versa
genders, ratings_shrek = Movies.rowElimWRT_ref_col('Gender identity (1 = female; 2 = male; 3 = self-described)','Shrek (2001)')

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
        choice = random.randint(0, 1)
        if choice:
            shrek_males.append(ratings_shrek[i])
        else:
            shrek_females.append(ratings_shrek[i])
    else: error('GENDER MISMATCH')
if count!=len(shrek_females)+len(shrek_males):error('POSSIBLE DATA MISMATCH')

Movies.plot(shrek_males, shrek_females, "Q3", n_bins = 3, titleX = "sample_ratings_male",titleY = "sample_ratings_female", \
    x1="Counts",y1="Counts",x2="Rating",y2="Rating")


#pval_q3 = stats.mannwhitneyu(shrek_males, shrek_females, alternative='two-sided', method='auto')
#print(pval_q3)
pval3 = Movies.utest2(shrek_males,shrek_females, hyp = 'two-sided', text = "that enjoyment of Shrek (2001) is gendered.")  


# What proportion of movies are rated differently by male and female viewers? 
Movies.proportion("gender blah")


ratings = np.array(data[:400])
count = 0
sig_diff = 0
for movie in range(len(ratings)):
    female_rating = []
    male_rating = []
    for i in range(len(genders)):
        if np.isnan(genders[i]) or np.isnan(ratings[movie][i]): continue
        if genders[i] == 1:
            female_rating.append(ratings[movie][i])
            count+=1
        elif genders[i] == 2:
            male_rating.append(ratings[movie][i])
            count+=1
        #due to scarcity of additional info on self-described individuals, gender randomly assigned for them
        elif genders[i] == 3:
            count+=1
            choice = random.randint(0, 1)
            if choice:
                female_rating.append(ratings[movie][i])
            else:
                male_rating.append(ratings[movie][i])
        else: error('GENDER MISMATCH')     
    if count!=len(female_rating)+len(male_rating):error('POSSIBLE DATA MISMATCH')

    if len(female_rating) or len(male_rating) <= 30:
        P = 20000
    else:
        P = 10000
    
    pval = stats.mannwhitneyu(female_rating, male_rating, alternative='two-sided', method='auto')

    if pval[1] <= 0.05:
        sig_diff+=1

prop_diff = sig_diff/len(ratings)
print(prop_diff)
# About ~31% of movies (124) show gendered preferences with an alpha 0.05

# Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings? 
idx_child = input.columns.get_loc('Are you an only child? (1: Yes; 0: No; -1: Did not respond)')
idx_ratings_LK = input.columns.get_loc('The Lion King (1994)')
LK_single = [] 
LK_multi = []
child_status = np.array(data[idx_child])
ratings_LK = np.array(data[idx_ratings_LK])
count = 0
for i in range(len(child_status)):
    if np.isnan(child_status[i]) or np.isnan(ratings_LK[i]): continue
    if child_status[i] == 1:
        LK_single.append(ratings_LK[i])
        count+=1
    elif child_status[i] == 0:
        LK_multi.append(ratings_LK[i])
        count+=1

if count!=len(LK_single)+len(LK_multi):error('POSSIBLE DATA MISMATCH')
pval_q5 = stats.mannwhitneyu(LK_single, LK_multi, alternative='greater', method='auto')
print(pval_q5)
#As pval >> 0.05 we fail to reject the null hypothesis; insifficient evidence to suggest single children 
# enjoy LK more than those with siblings

n_bins = 10
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].set_title("LK_single")
axs[1].set_title("LK_multi")
axs[0].set_ylabel('Counts')
axs[1].set_ylabel('Counts')
axs[0].set_xlabel('Avg Rating')
axs[1].set_xlabel('Avg Rating')
axs[0].hist(LK_single, bins=n_bins)
axs[1].hist(LK_multi, bins=n_bins)
fig.tight_layout()
fig.savefig('Q5.png', dpi=200) 
plt.show()

# What proportion of movies exhibit an “only child effect”, i.e. are rated different by viewers with siblings  vs. those without?  
count = 0
sig_diff = 0
for movie in range(len(ratings)):
    single_rating = []
    multi_rating = []
    for i in range(len(child_status)):
        if np.isnan(child_status[i]) or np.isnan(ratings[movie][i]): continue
        if child_status[i] == 1:
            single_rating.append(ratings[movie][i])
            count+=1
        elif child_status[i] == 0:
            multi_rating.append(ratings[movie][i])
            count+=1   

    if len(single_rating) or len(multi_rating) <= 30:
        P = 20000
    else:
        P = 10000

    pval = stats.mannwhitneyu(single_rating, multi_rating, alternative='two-sided', method='auto')

    if pval[1] <= 0.05:
        sig_diff+=1

prop_diff = sig_diff/len(ratings)
print(prop_diff)
# 10% of movies (40) are rated differently by viewers with siblings vs. those without with an alpha 0.05

# Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who prefer to watch them alone? 
idx_social = input.columns.get_loc('Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)')
idx_ratings_WW = input.columns.get_loc('The Wolf of Wall Street (2013)')
WW_alone = [] 
WW_ppl = []
group_status = np.array(data[idx_social])
ratings_WW = np.array(data[idx_ratings_WW])
count = 0
for i in range(len(group_status)):
    if np.isnan(group_status[i]) or np.isnan(ratings_WW[i]): continue
    if group_status[i] == 1:
        WW_alone.append(ratings_WW[i])
        count+=1
    elif child_status[i] == 0:
        WW_ppl.append(ratings_WW[i])
        count+=1

if count!=len(WW_alone)+len(WW_ppl):error('POSSIBLE DATA MISMATCH')
pval_q7 = stats.mannwhitneyu(WW_alone, WW_ppl, alternative='less', method='auto')
print(pval_q7)
#AS pval >> 0.05, fail to reject null hypothesis; insufficient evidence to suggest people who like to watch movies socially
#enjoy WW more than those who prefer to watch them alone

n_bins = 10
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].set_title("WW_alone")
axs[1].set_title("WW_ppl")
axs[0].set_ylabel('Counts')
axs[1].set_ylabel('Counts')
axs[0].set_xlabel('Rating')
axs[1].set_xlabel('Rating')
axs[0].hist(WW_alone, bins=n_bins)
axs[1].hist(WW_ppl, bins=n_bins)
fig.tight_layout()
fig.savefig('Q7.png', dpi=200) 
plt.show()

# What proportion of movies exhibit such a “social watching” effect? 
count = 0
sig_diff = 0
for movie in range(len(ratings)):
    alone_rating = []
    ppl_rating = []
    for i in range(len(group_status)):
        if np.isnan(group_status[i]) or np.isnan(ratings[movie][i]): continue
        if group_status[i] == 1:
            alone_rating.append(ratings[movie][i])
            count+=1
        elif group_status[i] == 0:
            ppl_rating.append(ratings[movie][i])
            count+=1   

    if len(alone_rating) or len(ppl_rating) <= 30:
        P = 20000
    else:
        P = 10000

    pval = stats.mannwhitneyu(alone_rating, ppl_rating, alternative='less', method='auto')
    if pval[1] <= 0.05:
        sig_diff+=1

prop_diff = sig_diff/len(ratings)
print(prop_diff)
# 7.75% of films (31) experience a social watching effect 


# Is the ratings distribution of ‘Home Alone (1990)’ different than that of ‘Finding Nemo (2003)’?  

idx_ratings_FN = input.columns.get_loc('Finding Nemo (2003)')
idx_ratings_HA = input.columns.get_loc('Home Alone (1990)')
# can use movies_clean here instead lol
ratings_FN = data[idx_ratings_FN] 
ratings_HA= data[idx_ratings_HA]
ratings_FN = ratings_FN[~np.isnan(ratings_FN)]
ratings_HA = ratings_HA[~np.isnan(ratings_HA)]

n_bins = 10
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].set_title("ratings_FN")
axs[1].set_title("ratings_HA")
axs[0].set_ylabel('Counts')
axs[1].set_ylabel('Counts')
axs[0].set_xlabel('Rating')
axs[1].set_xlabel('Rating')
axs[0].hist(ratings_FN, bins=n_bins)
axs[1].hist(ratings_HA, bins=n_bins)
fig.tight_layout()
fig.savefig('Q9.png', dpi=200) 
plt.show()

pval_q9 = stats.ks_2samp(ratings_FN, ratings_HA, alternative='two-sided', mode='auto')
print(pval_q9)
#As pval << 0.05 we reject the null hyp; there is sufficient evidence to claim that the twp distributions are indeed different


# There are ratings on movies from several franchises ([‘Star Wars’, ‘Harry Potter’, ‘The Matrix’, ‘Indiana  Jones’, ‘Jurassic Park’,  ‘Pirates of the Caribbean’, ‘Toy Story’, ‘Batman’]) in this dataset. 
# How many of these  are of inconsistent quality, as experienced by viewers? [Hint: You can use the keywords in quotation marks 
# featured in this question to identify the movies that are part of each franchise]  
#TO DO: regex - match blah blah Franchise Name blah blah and return all matches for movie in movie list
