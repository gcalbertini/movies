from os import error
import pandas as pd
import statistics as st
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import random

# read csv file
input = pd.read_csv("movieReplicationSet.csv")
df = input.to_numpy()
data = []
for col in range(df.shape[1]):
    vec = []
    for row in range(df.shape[0]):
        vec.append(df[row,col])
    data.append(vec)
data = np.array(data)

movies_clean = []
for entry in data: movies_clean.append(entry[~np.isnan(entry)])


movies = movies_clean[:400]
# Are movies that are more popular (operationalized as having more ratings) rated higher than movies that 
# are less popular? 

#Two sample t-test (unpaired or independent t-test)
#The two-sample (unpaired or independent) t-test compares the means of two independent groups,
#determining whether they are equal or significantly different. In two sample t-test, usually,
#we compute the sample means from two groups and derives the conclusion for the population’s
#means (unknown means) from which two groups are drawn.

#H_0: mean_populars = mean_sleepers
#H_1: mean_populars > mean_sleepers

#counting up non-NaN elements for each film; these are number of reviews given per problem statement--"popularity"
movies_popularities = [len(movies[i]) for i in range(len(movies))]
median_pop = st.median(movies_popularities)
populars = []
sleepers = []
#median-split of movie popularities
for i in range(len(movies_popularities)):
    if movies_popularities[i] > median_pop:
        populars.append(movies_clean[i])
    elif movies_popularities[i] < median_pop:
        sleepers.append(movies_clean[i])

    if movies_popularities[i] == median_pop:
        choice = random.randint(0, 1)
        if choice:
            populars.append(movies_clean[i])
        else:
            sleepers.append(movies_clean[i])
    
if (len(sleepers)+len(populars)!=400):error("DATA MISMATCH")
            

sample_means_populars = []
sample_means_sleepers = []
for movie in populars: sample_means_populars.append(np.mean(movie))
for movie in sleepers: sample_means_sleepers.append(np.mean(movie))
variances1 = [st.variance(sample_means_populars), st.variance(sample_means_sleepers)]

n_bins = 13
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].set_title("sample_means_populars")
axs[1].set_title("sample_means_sleepers")
axs[0].set_ylabel('Counts')
axs[1].set_ylabel('Counts')
axs[0].set_xlabel('Avg Rating')
axs[1].set_xlabel('Avg Rating')
axs[0].hist(sample_means_populars, bins=n_bins)
axs[1].hist(sample_means_sleepers, bins=n_bins)
med1 = np.median(sample_means_populars)
med2 = np.median(sample_means_sleepers)
plt.show()


#TO-DO: ASSUMPTIONS for U TEST

pval_q1 = stats.mannwhitneyu(sample_means_populars, sample_means_sleepers, alternative='greater', method='auto')
print(pval_q1)

#As pval << 0.05 we reject the null hypothesis. There is sufficient evidence to suggest that movies that are more
#popular have ratings that are higher than movies that are less popular.

# Are movies that are newer rated differently than movies that are older? 
movie_yrs = input.iloc[0,:400].to_string()
entries = movie_yrs.split('\n')
years = []
for movie in entries:
    years.append(movie[movie.find("(")+1:movie.find(")")])
years[6] = '1985' #Rambo: First Blood Part II missing date
years = list(map(int, years))
mean_yrs = np.mean(years)

sample_means_movies = []
for i in range(len(years)): sample_means_movies.append(np.mean(movies_clean[i]))
#movies_yr_rating = []
#for i in range(len(years)): movies_yr_rating.append([years[i],sample_means_movies[i]])

#median-split of movie years
newer = []
older = []
median_age = st.median(years)
for i in range(len(years)):
    if years[i] > median_age:
        newer.append(sample_means_movies[i])
    elif years[i] < median_age:
        older.append(sample_means_movies[i])

    if years[i] == median_age:
        choice = random.randint(0, 1)
        if choice:
            newer.append(sample_means_movies[i])
        else:
            older.append(sample_means_movies[i])

if (len(newer)+len(older)!=400):error("DATA MISMATCH")

n_bins = 13
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].set_title("sample_means_older")
axs[1].set_title("sample_means_newer")
axs[0].set_ylabel('Counts')
axs[1].set_ylabel('Counts')
axs[0].set_xlabel('Avg Rating')
axs[1].set_xlabel('Avg Rating')
axs[0].hist(older, bins=n_bins)
axs[1].hist(newer, bins=n_bins)
plt.show()

#TO DO: ASSUMPTIONS for U
variances2 = [st.variance(newer), st.variance(older)]

pval_q2 = stats.mannwhitneyu(newer, older, alternative='two-sided', method='auto')
print(pval_q2)       

#As pval > 0.05 we fail to reject the null hypothesis. There is insufficient evidence to suggest that movies 
#that are newer are rated differently than movies that are older.

# Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently? 

idx_gender = input.columns.get_loc('Gender identity (1 = female; 2 = male; 3 = self-described)')
idx_ratings_shrek = input.columns.get_loc('Shrek (2001)')
shrek_males = [] 
shrek_females = []
genders = np.array(data[idx_gender])
ratings_shrek = np.array(data[idx_ratings_shrek])
count = 0
for i in range(len(genders)):
    if np.isnan(genders[i]) or np.isnan(ratings_shrek[i]): continue
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

n_bins = 5
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].set_title("sample_ratings_male")
axs[1].set_title("sample_ratings_female")
axs[0].set_ylabel('Counts')
axs[1].set_ylabel('Counts')
axs[0].set_xlabel('Avg Rating')
axs[1].set_xlabel('Avg Rating')
axs[0].hist(shrek_males, bins=n_bins)
axs[1].hist(shrek_females, bins=n_bins)
plt.show()

variances3 = [st.variance(shrek_males), st.variance(shrek_females)]

#TO-DO: ASSUMPTIONS
pval_q3 = stats.mannwhitneyu(sample_means_populars, sample_means_sleepers, alternative='two-sided', method='auto')

print(pval_q3)

#As pval > 0.05 we fail to reject the null hypothesis. There is insufficient evidence to suggest that enjoyment
#of Shrek (2001) is gendered.

# What proportion of movies are rated differently by male and female viewers? 
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
# About ~XX% of movies show gendered preferences with an alpha 0.05

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

    pval_q1 = stats.mannwhitneyu(single_rating, multi_rating, alternative='two-sided', method='auto')


    if pval[1] <= 0.05:
        sig_diff+=1

prop_diff = sig_diff/len(ratings)
print(prop_diff)
# About ~30% of movies show an "only child effect" with an alpha 0.05

# Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who  prefer to watch them alone? 
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
# About % of films expereince a social watching effect 


# Is the ratings distribution of ‘Home Alone (1990)’ different than that of ‘Finding Nemo (2003)’?  
# There are ratings on movies from several franchises ([‘Star Wars’, ‘Harry Potter’, ‘The Matrix’, ‘Indiana  Jones’, ‘Jurassic Park’,  ‘Pirates of the Caribbean’, ‘Toy Story’, ‘Batman’]) in this dataset. 
# How many of these  are of inconsistent quality, as experienced by viewers? [Hint: You can use the keywords in quotation marks 
# featured in this question to identify the movies that are part of each franchise]  