#!/usr/bin/env python
# coding: utf-8

# ## Import Datasets

# In[1]:


import pandas as pd
import seaborn as sns
import pickle
import numpy as np


# In[2]:


subscribers = pd.read_pickle(r'Data/subscribers')


# In[4]:


subscribers['subid'].nunique()


# In[5]:


engagement = pd.read_pickle(r'Data/engagement')


# In[6]:


engagement.isnull().sum()


# In[7]:


engagement_nona = engagement.dropna()


# In[13]:


#Getting correlation stats of engagement
people = engagement_nona.columns.to_list()
people.remove('subid')
people.remove('date')
people.remove('payment_period')
list_of_pairs = [(p1, p2) for p1 in people for p2 in people if p1 != p2]


# In[14]:


from scipy.stats import pearsonr

def get_corr(t1):
    print(t1[0])
    print(t1[1])
    return pearsonr(engagement_nona[t1[0]],engagement_nona[t1[1]])


# In[15]:


for i in (list_of_pairs):
    print(get_corr(i))
    

# correlation check seems okay


# In[16]:


engagement_nona['subid'].nunique()


# In[10]:


cs = pd.read_pickle(r'Data/customer_service_reps')


# In[12]:


cs['subid'].nunique()


# In[13]:


cs.shape


# In[68]:


min(cs['account_creation_date'])


# In[14]:


#Get list of  customer IDs that are present in all 3 datasets

engagement_userids = engagement['subid'].tolist()
subscribers_userids = subscribers['subid'].tolist()
customers_userids = cs['subid'].tolist()

len(list(set(engagement_userids) & set(subscribers_userids)))

customer_list = list(set(customers_userids) & set(subscribers_userids) & set(engagement_userids))


# # AB Testing
# - No Trial Fee vs Discounted Trial Fee
# - 7 Day Trial Period vs 14 Day Trial Period 
# 
# 
# Control:
# Base_UAE_14_Day Trial
# 
# Treatment:
# High_UAE_14_Day_Trial 
# Low_UAE_No_Trial

# Target variable: Cancel before trial ends - If they convert to full time subscribers
#                 Refund after trial ends - No revenue from them too

# In[15]:


subscribers['plan_type'].value_counts()


# In[16]:


#Check customers' earliest account creation date
min(subscribers['account_creation_date'])


# In[522]:


#Get a pared down dataframe with relevant variables
df = subscribers[['plan_type','account_creation_date','cancel_before_trial_end','refund_after_trial_TF','retarget_TF']]

df['month'] = df['account_creation_date'].dt.month
df['year'] = df['account_creation_date'].dt.year

df = df[df['plan_type'].isin(['base_uae_14_day_trial','low_uae_no_trial','high_uae_14_day_trial'])]


# In[526]:


df.groupby(['year','month','retarget_TF','plan_type']).count()


# ## Part 1 - Hypo Test

# In[545]:


#High UAE 14 Day Trial

#This plan was only offered in November 2019, so lets compare the conversions for that time period

df_base = df[df['plan_type'] == 'base_uae_14_day_trial']
df_base_nov = df_base[(df_base['year'] == 2019) & (df_base['month'] == 11)]
df_high = df[df['plan_type'] == 'high_uae_14_day_trial']

base_conv_rate = len(df_base_nov[(df_base_nov['cancel_before_trial_end'] == True) & (df_base_nov['refund_after_trial_TF'] == False)])/len(df_base_nov)
high_conv_rate = len(df_high[(df_high['cancel_before_trial_end'] == True) & (df_high['refund_after_trial_TF'] == False)])/len(df_high)

n = len(df_high)

print(base_conv_rate)
print(high_conv_rate)
print(n)


# In[534]:


import math
#Calculate Z-score
z = (high_conv_rate - base_conv_rate) / math.sqrt((base_conv_rate * (1- base_conv_rate))/n)
print(z)


# In[351]:


#low UAE 14 Day Trial

df_base = df[df['plan_type'] == 'base_uae_14_day_trial']
#Lets get only the base results for the same time period during which low trial period was offered

df_base_time = df_base[(df_base['year'] == 2019) & df_base['month'].isin([7,8,9,10,11])]

df_low = df[df['plan_type'] == 'low_uae_no_trial']

base_conv_rate = len(df_base_time[(df_base_time['cancel_before_trial_end'] == True) & (df_base_time['refund_after_trial_TF'] == False)])/len(df_base_time)
low_conv_rate = len(df_low[(df_low['cancel_before_trial_end'] == True) & (df_low['refund_after_trial_TF'] == False)])/len(df_low)

n = len(df_low)

print(low_conv_rate)
print(base_conv_rate)
print(n)


# In[27]:


import math
#Calculate Z-score
z = (low_conv_rate - base_conv_rate) / math.sqrt((base_conv_rate * (1- base_conv_rate))/n)
print(z)


# In[548]:


#For low trial, we realise that the entire population had been retargeted. So lets adjust our statistics slightly, to compare the conversion rates of 
#those who are retargeted, which plan is best for retargeting

#low UAE 14 Day Trial

df_base = df[df['plan_type'] == 'base_uae_14_day_trial']
#Lets get only the base results for the same time period during which low trial period was offered, AND those who were retargeted

df_base_time = df_base[(df_base['year'] == 2019) & df_base['month'].isin([7,8,9,10,11])]
df_base_retargeted = df_base_time[df_base_time['retarget_TF'] == True]

df_low = df[df['plan_type'] == 'low_uae_no_trial']

base_conv_rate = len(df_base_retargeted[(df_base_retargeted['cancel_before_trial_end'] == True) & (df_base_retargeted['refund_after_trial_TF'] == False)])/len(df_base_retargeted)
low_conv_rate = len(df_low[(df_low['cancel_before_trial_end'] == True) & (df_low['refund_after_trial_TF'] == False)])/len(df_low)

n = len(df_low)

print(low_conv_rate)
print(base_conv_rate)


# In[46]:


import math
#Calculate Z-score
z = (low_conv_rate - base_conv_rate) / math.sqrt((base_conv_rate * (1- base_conv_rate))/n)
print(z)


# But we need to check whether both tests had enough power for the test to be significant.

# ## Part 2 - Optimal Sample Size

# In[106]:


#High Plan

t_alpha = 1.96
p_bar = (high_conv_rate + base_conv_rate) / 2
p0 = base_conv_rate
p1 = high_conv_rate
delta = (high_conv_rate - base_conv_rate)
t_beta = 0.842

optimal = (t_alpha * math.sqrt((2*p_bar*(1-p_bar))) + t_beta * math.sqrt(p0*(1-p0) + p1*(1-p1)))**2 * (1/(delta**2))
print(optimal)


# In[110]:


#Low

t_alpha = 1.96
p_bar = (low_conv_rate + base_conv_rate) / 2
p0 = base_conv_rate
p1 = low_conv_rate
delta = (low_conv_rate - base_conv_rate)
t_beta = 0.842

optimal = (t_alpha * math.sqrt((2*p_bar*(1-p_bar))) + t_beta * math.sqrt(p0*(1-p0) + p1*(1-p1)))**2 * (1/(delta**2))
print(optimal)


# For High_UAE_14_Day_Trial, plan hurt overall conversion, and the test was underpowered to be able to detect the difference. 
# 
# For the Low_UAE_no_trial, plan increased conversion, and the test had enough power to detect the difference. 

# But we should also check that the sampling was representative, and there was no confounding factors. 

# Observations from Tableau:
#     
# No Trial
# - All were from Organic Search, but most of the base people were from Facebook
# - Male to Female Ratio was fairly similar, ~5% more females in low
# - Monthly Price for low is 1.0643, no join fee, Base monthly price is 4.7343
# - 100% of Low_UAE_no_Trial are "retargeted"; base 97% are not
# - Package Type Null
# 
# 
# High Trial
# - Attribution proportions look similar
# - A high proportion of ppl selected "Replace OTT" as their intended use
# - Male to Female Ratio fairly similar
# - Monthly Price is 5.1013
# -

# # Advertising Channel Spend 
# 
# - Use attribution (Technical)
# - Use attribution (Survey)
# 
# Calculate CPA and CAC for each Channel

# ## Part 1: Get % of ad Spend based on proportion of data we have

# In[259]:


#Create month, year variables in customer service dataset
cs['month'],cs['year'] = cs['account_creation_date'].dt.month, cs['account_creation_date'].dt.year


# In[260]:


cs_nodupes = cs.drop_duplicates(subset=['subid'])


# In[261]:


cs_count = pd.DataFrame(cs_nodupes.groupby(['year','month'])['account_creation_date'].count())

cs_count.reset_index(inplace=True)


# In[263]:


cs_nodupes['subid'].nunique()


# In[264]:


#Create month, year variables in subscribers dataset
subscribers['month'],subscribers['year'] = subscribers['account_creation_date'].dt.month, subscribers['account_creation_date'].dt.year


# In[265]:


subscribers_count = pd.DataFrame(subscribers.groupby(['year','month'])['account_creation_date'].count())

subscribers_count.reset_index(inplace=True)


# In[267]:


#merge these two groupbys to get number of account created in each month, for both subscribers and CS dataset

total = cs_count.merge(subscribers_count,on=['year','month'])

total['percent'] = total['account_creation_date_y'] / total['account_creation_date_x']


# In[269]:


total


# In[270]:


#On an aggregate level
subscribers['subid'].nunique() / cs_nodupes['subid'].nunique()


# In[271]:


#Since we only know 16% of data, we should pare the spending down by roughly 16.6% as a whole


# In[272]:


ad_spend = pd.read_csv('Data/ad_spend.csv',thousands=',')


# In[273]:


#Adjust datetime variables

ad_spend['year'] = pd.to_datetime(ad_spend['date']).dt.year
ad_spend['month'] = pd.to_datetime(ad_spend['date']).dt.month

ad_spend.drop('date',axis=1,inplace=True)

ad_spend = ad_spend.set_index(['year','month'])

ad_spend.reset_index(inplace=True)


# In[276]:


ad_spend


# In[278]:


#Take % of customers we know the data for, for each month * the respective attribution columns

ad_spend_paredown = ad_spend.iloc[:,:2].join(ad_spend.iloc[:,2:].mul(total['percent'],axis=0))

ad_spend_paredown.set_index(['year','month'],inplace=True)


# In[279]:


ad_spend_paredown


# In[280]:


ad_spend_new = pd.DataFrame(ad_spend_paredown.stack()).reset_index()
ad_spend_new.columns = ['year','month','attribution','spend']


# In[281]:


ad_spend_new


# ## Part 2: Get CPA and CAC

# In[282]:


#Keep only subscribers that we have spend data on

ad_channels = subscribers[subscribers['attribution_technical'].isin(ad_spend.columns.tolist())]

ad_channels['year'] = ad_channels['account_creation_date'].dt.year
ad_channels['month'] = ad_channels['account_creation_date'].dt.month


# In[284]:


#How paying users are defined
paying_users = ad_channels[(ad_channels['cancel_before_trial_end'] == True) & (ad_channels['refund_after_trial_TF'] == False)]

paying_users.shape


# In[285]:


total_users_counts = pd.DataFrame(ad_channels.groupby(['year','month','attribution_technical'])['subid'].count()).reset_index()

paying_users_counts = pd.DataFrame(paying_users.groupby(['year','month','attribution_technical'])['subid'].count()).reset_index()

total_users_counts.columns = ['year','month','attribution','total_users']
paying_users_counts.columns = ['year','month','attribution','paying_users']


# In[288]:


merged_users = total_users_counts.merge(paying_users_counts, on=['year','month','attribution'])


# In[289]:


merged_users


# In[290]:


#Merge the number of users with the respective spend data
ad_merged = merged_users.merge(ad_spend_new, left_on=['year','month','attribution'],right_on=['year','month','attribution'],how='right')


# In[291]:


ad_merged['CPA'] = ad_merged['spend'] / ad_merged['total_users']
ad_merged['CAC'] = ad_merged['spend'] / ad_merged['paying_users']


# Exclude June Data because the earliest account_creation_date was June 30th, so there was only 2 days to sign up, whereas the rest of the months had the whole month

# In[293]:


ad_merged_filtered = ad_merged[ad_merged['month'] != 6]


# In[294]:


ad_merged_filtered


# ## Repeat for Attribution_Survey

# In[381]:


subscribers['attribution_survey'].isna().sum() / len(subscribers) #1% of missing data, dont need to pare down


# In[382]:


ad_channels = subscribers[subscribers['attribution_survey'].isin(ad_spend.columns.tolist())]

ad_channels['year'] = ad_channels['account_creation_date'].dt.year
ad_channels['month'] = ad_channels['account_creation_date'].dt.month

ad_channels.shape


# In[384]:


paying_users = ad_channels[(ad_channels['cancel_before_trial_end'] == True) & (ad_channels['refund_after_trial_TF'] == False)]

paying_users.shape


# In[386]:


total_users_counts = pd.DataFrame(ad_channels.groupby(['year','month','attribution_technical'])['subid'].count()).reset_index()

paying_users_counts = pd.DataFrame(paying_users.groupby(['year','month','attribution_technical'])['subid'].count()).reset_index()

total_users_counts.columns = ['year','month','attribution','total_users']

paying_users_counts.columns = ['year','month','attribution','paying_users']


# In[387]:


merged_users = total_users_counts.merge(paying_users_counts, on=['year','month','attribution'])


# In[389]:


ad_merged = merged_users.merge(ad_spend_new, left_on=['year','month','attribution'],right_on=['year','month','attribution'],how='right')


# In[390]:


ad_merged['CPA'] = ad_merged['spend'] / ad_merged['total_users']
ad_merged['CAC'] = ad_merged['spend'] / ad_merged['paying_users']


# In[391]:


ad_merged.head()


# Exclude June Data because the earliest account_creation_date was June 30th, so there was only 2 days to sign up, whereas the rest of the months had the whole month

# In[392]:


ad_merged_survey = ad_merged[ad_merged['month'] != 6]
ad_merged_survey.to_csv('ad_merged_survey_CAC_CPA.csv')


# In[393]:


ad_merged_survey


# ## Get Marginal CAC

# Tried looking at Marginal CAC across time periods, but analysis was not fruitful, and spend increased/decreased over time. Clear insights could not be made, so analysis was excluded from final presentation.

# In[358]:


ad_filtered_pivot = pd.pivot_table(ad_merged_filtered, index = 'attribution',columns=['year','month'])[['spend','subid']]


# In[359]:


ad_filtered_pivot


# In[211]:


#Get marginal spend and CPA for specified month,year and channel
def get_marginal_CPA(year,month,channel):
    if year == 2019:
        marginal_spend = ad_filtered_pivot.loc[channel,('spend',year,month)] -  ad_filtered_pivot.loc[channel,('spend',year,month-1)]
        marginal_conv = ad_filtered_pivot.loc[channel,('subid',year,month)] -  ad_filtered_pivot.loc[channel,('subid',year,month-1)]
    if year == 2020:
        if month == 1:
            marginal_spend = ad_filtered_pivot.loc[channel,('spend',year,month)] -  ad_filtered_pivot.loc[channel,('spend',2019,12)]
            marginal_conv = ad_filtered_pivot.loc[channel,('subid',year,month)] -  ad_filtered_pivot.loc[channel,('subid',2019,12)]
        else:
            marginal_spend = ad_filtered_pivot.loc[channel,('spend',year,month)] -  ad_filtered_pivot.loc[channel,('spend',year,month-1)]
            marginal_conv = ad_filtered_pivot.loc[channel,('subid',year,month)] -  ad_filtered_pivot.loc[channel,('subid',year,month-1)]
   
    return marginal_spend,marginal_conv


# In[213]:


ad_filtered_pivot['marginal_spend'], ad_filtered_pivot['marginal_conv'] = get_marginal_CPA(2019,8,'email')


# In[214]:


ad_filtered_pivot


# # Churn Modelling
# 
# Model behavior from Payment Period 1 -> 2

# ## Variable Prep and Merging

# In[636]:


#Filter for customers in the customer service dataset that is in the intersection of all 3 datasets
cs_all = cs[cs['subid'].isin(customer_list)]

cs_all.shape


# In[20]:


#Filter for customers in/was in payment period 1
cs_pp1 = cs_all[cs_all['payment_period'] == 1]

sum(cs_pp1['renew'].isna())


# In[86]:


#check value counts of 'renew' variable
cs_pp1['renew'].value_counts()


# In[84]:


cs_pp1.shape


# In[25]:


#Get target variable 'churn' using the below method

def get_churn(row):
    if row.renew == False: #Customer cancelled
        label = 1
    elif row.current_sub_TF == False: #Customer cannot access current content = considered churn
        label = 1
    elif row.revenue_net_1month <= 0: #Customer got a refund, so net revenue <=0 = considered churn
        label = 1
    else:
        label = 0
    return label

cs_pp1['churn'] = cs_pp1.apply(get_churn,axis=1)


# In[26]:


#Filter for customers who were truly in payment_period 1, they didnt drop off during trial or got a refund
cs_stayed = cs_pp1[(cs_pp1['trial_completed_TF'] == True)]
cs_stayed = cs_stayed[cs_stayed['revenue_net_1month'] > 0]


# In[28]:


cs_stayed.shape


# In[29]:


cs_stayed['churn'].isna().sum() #Check that no missing values in 'churn'


# In[30]:


cs_stayed.head()


# ### Merge Engagement into Dataset with 'Churn' Variable

# In[31]:


#Get a pared down version of previous dataset
current_df = cs_stayed[['subid','account_creation_date','revenue_net_1month','payment_period','renew','churn']]


# In[32]:


current_df.shape


# In[33]:


#Get engagement data for these customers
engagement_all = engagement[engagement['subid'].isin(customer_list)].dropna()


# In[36]:


#Here I split engagement data into during trial period, and during period 1

eng_data = engagement_all.groupby(['subid','payment_period']).sum().reset_index()


# In[38]:


trial_engagement = eng_data[eng_data['payment_period'] == 0]

#Rename columns to include 'trial'
trial_engagement.columns = ['subid','payment_period_trial','app_opens_trial','cust_service_mssgs_trial','num_videos_completed_trial','num_videos_more_than_30_seconds_trial','num_videos_rated_trial','num_series_started_trial']

trial_engagement


# In[401]:


#Merge trial engagement data with working dataframe
current_df_eng = current_df.merge(trial_engagement,on='subid',how='left')


# In[410]:


#Payment period 1 engagement Data; scared if just take sum, might run into data leakage problem, hence normalize by daily
#Here daily is calculated by days between last engagement date in pp1 - first engagement date in pp1 
pp1_eng = engagement_all[engagement_all['payment_period'] != 0].groupby('subid').agg({'date':['min','max'],'app_opens': 'sum','cust_service_mssgs':'sum','num_videos_completed':'sum','num_videos_more_than_30_seconds':'sum','num_videos_rated':'sum','num_series_started':'sum'}).reset_index()

#Here by inspecting the dates, we found out that the 'payment_period' variable is diff from that in cust service rep dataset. This is in months, not in 4 months.

pp1_eng.columns = ['subid','date_min','date_max','app_opens_sum','cust_service_mssgs_sum','num_videos_completed_sum','num_videos_more_than_30_seconds_sum','num_videos_rated_sum','num_series_started_sum']


# In[412]:


pp1_eng['days'] = (pp1_eng['date_max'] - pp1_eng['date_min']).dt.days


# In[423]:


pp1_eng


# In[425]:


pp1_eng['days'] = pp1_eng['days'].replace(0,1) #Because '0' days create a problem when later eng data / days, we replace '0' days with '1' day


# In[427]:


#Get daily engagement variables

col_names = pp1_eng.columns.to_list()[3:-1]

for i in col_names:
    new_label = i + '_daily'
    pp1_eng[new_label] = pp1_eng[i] / pp1_eng['days']

pp1_eng = pp1_eng.drop(col_names,axis=1)
pp1_eng = pp1_eng.drop(['date_min','date_max','days'],axis=1)


# In[428]:


#Merge pp1 engagement data with working dataframe
current_df_alleng = current_df_eng.merge(pp1_eng,on='subid', how='left')


# In[429]:


current_df_alleng.isnull().sum()


# In[431]:


#In this case, since missing data means no engagement, we can just fill na = 0

current_df_no_na = current_df_alleng.fillna(0)


# ### Merged cust serv with engagement, next merge with subscribers dataset

# In[432]:


#Get subscribers dataset for those in intersection of all 3 datasets
subscribers_all = subscribers[subscribers['subid'].isin(customer_list)]


# In[435]:


#Merge previous dataset with this subscribers dataset, on subid
full_df = current_df_no_na.merge(subscribers_all,on='subid',how='left')


# In[438]:


full_df.shape


# In[554]:


#Drop columns that we will not need for churn modelling
full_df_colsdropped = full_df.drop(['renew','months_per_bill_period','payment_period_trial','creation_until_cancel_days','cancel_before_trial_end','paid_TF','initial_credit_card_declined','trial_end_date','account_creation_date_y','retarget_TF','country','language'],axis=1)


full_df_colsdropped = full_df_colsdropped.drop(['revenue_net','attribution_survey','age','revenue_net_1month'],axis=1)


# In[555]:


full_df_colsdropped.columns


# In[441]:


full_df_colsdropped['churn'].value_counts() #Check proportion of churn


# In[442]:


full_df_colsdropped.isnull().sum() #Check which columns have null values


# In[556]:


#Drop self-reported columns that have too many NA Values
full_df_drop_selfreported = full_df_colsdropped.drop(['num_weekly_services_utilized','weekly_consumption_hour','num_ideal_streaming_services'],axis=1)


# In[560]:


#Impute the missing values for package_type, preferred_genre, and intended_use, and op_sys

import random
import numpy as np

full_df_drop_selfreported['preferred_genre'].value_counts(normalize=True)

full_df_drop_selfreported['preferred_genre'] = full_df_drop_selfreported['preferred_genre'].fillna(pd.Series(np.random.choice(['comedy', 'drama', 'regional','international','other'], 
                                                      p=[0.65, 0.24, 0.052,0.033,0.025], size=len(full_df_drop_selfreported))))

full_df_drop_selfreported['intended_use'].value_counts(normalize=True)

full_df_drop_selfreported['intended_use'] = full_df_drop_selfreported['intended_use'].fillna(pd.Series(np.random.choice(['access to exclusive content', 'replace OTT', 'supplement OTT','expand regional access','expand international access','education','other'], 
                                                      p=[0.40, 0.28, 0.11,0.085,0.071,0.027,0.027], size=len(full_df_drop_selfreported))))

full_df_drop_selfreported['op_sys'].fillna('iOS',inplace=True)


#Assume here that those who do not have package_type are people who are not customers of our internet package
full_df_drop_selfreported['package_type'].fillna('no_package',inplace=True)


# In[561]:


full_df_drop_selfreported.isna().sum() #No more NA values (except for join_fee, but this will be dropped later)


# In[562]:


#Filter out those who had refund after trial TF flagged as True
full_df_drop_selfreported = full_df_drop_selfreported[full_df_drop_selfreported['refund_after_trial_TF'] == False]


# In[448]:


full_df_drop_selfreported.shape


# In[563]:


full_df_drop_selfreported.to_csv('full_df_drop_selfreported_pp1.csv') #save to .csv to save progress


# ## Modelling

# In[564]:


df = pd.read_csv('full_df_drop_selfreported_pp1.csv',index_col=0) #import previous .csv file


# In[455]:


df['churn'].value_counts(normalize=True)


# In[456]:


df.shape


# In[566]:


#Because the other plans were making too many dummy variables that increased dimensionality, only keep those observations that formed majority of people
df = df[df['plan_type'].isin(['base_uae_14_day_trial','high_uae_14_day_trial','low_uae_no_trial'])]


# In[458]:


#this dropped 13 people - not very significant
df.shape


# In[459]:


#Organize into X and Y variables

#Here I choose to drop attribution technical bc it creates too many dimensions
list_cols = df.columns.to_list()
remove_list = ['subid','churn','join_fee','payment_period', 'attribution_technical','month','year','refund_after_trial_TF','monthly_price','discounted_price','account_creation_date_x']
list_x = [col for col in list_cols if col not in remove_list]

X = df[list_x]
Y = df["churn"]


# In[462]:


# First start with dealing with categorical variables

#Dummy variables
list_of_cols = ['package_type','preferred_genre','intended_use','male_TF','op_sys','plan_type','payment_type']

for var in list_of_cols:
    cat_list = pd.get_dummies(X[var],drop_first=True,prefix=var)
    X = X.join(cat_list,lsuffix='_left')

X = X.drop(list_of_cols,axis=1)


# In[463]:


X.columns


# In[465]:


from sklearn.model_selection import train_test_split

#Train test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = 5)


# ### Building Models - Decision Tree first

# In[466]:


#Build Models

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


# In[510]:


tree_1 = DecisionTreeClassifier(criterion='entropy',max_depth = 6, max_features = 12,min_samples_leaf=30)
tree_1.fit(X_train,Y_train)

print("Accuracy for Decision Tree Training set: %.5f" %(tree_1.score(X_train,Y_train)))
print("Accuracy for Decision Tree Test set: %.5f" %(tree_1.score(X_test,Y_test)))

print(confusion_matrix(Y_test,tree_1.predict(X_test)))
print(classification_report(Y_test,tree_1.predict(X_test)))
print(f1_score(Y_test,tree_1.predict(X_test)))


# In[468]:


#Get gridsearch to find best parameters to fit into model

from sklearn.model_selection import GridSearchCV

param_test = {
        'max_depth': [2,3,4,5,6], 'max_features': [6,8,10,12]
    }

estimator = DecisionTreeClassifier(random_state = 0,min_samples_leaf=30)
gsearch = GridSearchCV(estimator, param_grid = param_test, cv=10)
gsearch.fit(X_train,Y_train)
print(gsearch.best_params_)


# In[472]:


#Adjust Threshold to check results 

#Set threshold
y_pred = (tree_1.predict_proba(X_test)[:,1] >= 0.55).astype(bool)

#Metrics
print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))

#F1 Score
print(f1_score(Y_test, y_pred))


# In[473]:


#Get feature importance of each variable

columns = X.columns
for i in range(len(columns)):
    print(columns[i])
    print(tree_1.feature_importances_[i])


# In[153]:


#Plot tree

from IPython.display import SVG, display, Image
from graphviz import Source

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'


# In[474]:


graph = Source(tree.export_graphviz(tree_1, out_file=None
   , feature_names=X.columns, class_names=['No Churn', 'Churn'] 
   , filled = True))
display(SVG(graph.pipe(format='svg')))


# In[476]:


graph.render(filename='churn_tree_2')


# ### Try with more robust DT methods

# In[479]:


#Creating random forest models

from sklearn.ensemble import RandomForestClassifier

rf_1 =RandomForestClassifier(n_estimators=20,criterion="entropy",max_depth=8,max_features=12,min_samples_leaf=30)
rf_1.fit(X_train,Y_train)

print("Accuracy for Random Forest Training set: %.5f" %(rf_1.score(X_train,Y_train)))
print("Accuracy for Random Forest Test set: %.5f" %(rf_1.score(X_test,Y_test)))

print(confusion_matrix(Y_test,rf_1.predict(X_test)))
print(classification_report(Y_test,rf_1.predict(X_test)))


# In[478]:


param_test = {
        'max_depth': [4,5,6,7,8], 'max_features': [6,8,10,12]
    }

estimator = RandomForestClassifier(random_state = 0,n_estimators=10)
gsearch = GridSearchCV(estimator, param_grid = param_test, cv=10)
gsearch.fit(X_train,Y_train)
print(gsearch.best_params_)


# In[480]:


#Get Feature Importance

columns = X_train.columns
for i in range(len(columns)):
    print(columns[i])
    print(rf_1.feature_importances_[i])


# In[481]:


from sklearn.ensemble import GradientBoostingClassifier

gb_1 =GradientBoostingClassifier(n_estimators=50,max_depth=4,min_samples_leaf=30,max_features=8)
gb_1.fit(X_train,Y_train)

print("Accuracy for RF Training set: %.5f" %(gb_1.score(X_train,Y_train)))
print("Accuracy for RF Test set: %.5f" %(gb_1.score(X_test,Y_test)))

print(confusion_matrix(Y_train,gb_1.predict(X_train)))
print(classification_report(Y_test,gb_1.predict(X_test)))


# ### Logistic Regression

# In[638]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(C=100)
lg.fit(X_train, Y_train)
y_pred = lg.predict(X_test)

print("Accuracy for Log Reg Training set: %.5f" %(lg.score(X_train,Y_train)))
print("Accuracy for Log Reg Tree Test set: %.5f" %(lg.score(X_test,Y_test)))

print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
print(f1_score(Y_test,y_pred))


# In[486]:


index = 0
for i in lg.coef_[0]:
    print("%s      %.5f" %(X_train.columns[index],i))
    index += 1


# ### OLS 

# In[639]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)

print("Accuracy for Linear Reg Training set: %.5f" %(lr.score(X_train,Y_train)))
print("Accuracy for Linear Reg Tree Test set: %.5f" %(lr.score(X_test,Y_test)))


# ## Evaluation : Building AUC, ROC Curves

# In[488]:


#Method for ROC Curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def plot_roc_curve(predict1,predict2):
    logit_roc_auc = roc_auc_score(Y_test, predict1)
    fpr, tpr, thresholds = roc_curve(Y_test, predict2)
    plt.figure()
    plt.plot(fpr, tpr, label='Area = %0.3f' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()


# In[489]:


#Tree ROC
plot_roc_curve(tree_1.predict(X_test),tree_1.predict_proba(X_test)[:,1])


# In[490]:


#Log Reg ROC
plot_roc_curve(lg.predict(X_test),lg.predict_proba(X_test)[:,1])


# In[491]:


#Random Forest ROC
plot_roc_curve(rf_1.predict(X_test),rf_1.predict_proba(X_test)[:,1])


# In[492]:


#GBDT ROC
plot_roc_curve(gb_1.predict(X_test),gb_1.predict_proba(X_test)[:,1])


# # Revenue Modelling

# In[602]:


#Get dataframe of predicted probabilities from decision tree model, join with actual Y labels, merge subids back to this dataframe

Churn_proba_testdata = pd.DataFrame(data=tree_1.predict_proba(X_test)).join(pd.DataFrame(Y_test).reset_index())

Churn_proba_testdata = Churn_proba_testdata.merge(pd.DataFrame(df['subid']).reset_index(),on='index')


# In[603]:


Churn_proba_testdata 


# In[604]:


#Organize and rename columns
churn_df = Churn_proba_testdata[['index','subid','churn',1]]
churn_df.columns = ['index','subid','y_actual','y_pred']
churn_df


# In[495]:


#hardcoded variable
accept_offer_rate = 0.3


# In[496]:


#Individual methods

def get_offer(y_pred,threshold):
    if y_pred >= threshold:
        return 1
    else:
        return 0

import random

def accept_offer(get_offer):
    if get_offer == 1:
        if random.random() <= accept_offer_rate:
            return 1
        else:
            return 0
    else:
        return 0
    
def renew_at_base(row,threshold):
    label = 0
    label_1 = 'get_offer_' + str(threshold)
    label_2 = 'accept_offer_' + str(threshold)
    #if they were going to renew at base, and if they didnt get an offer, or they got the offer but didnt accept:
    if (row.y_actual == 0):
        if (row[label_1] == 0) | ((row[label_1] == 1) & (row[label_2] == 0)):
            label = 1
    return label

def get_plan(row,threshold):
    label_1 = 'accept_offer_' + str(threshold)
    label_2 = 'renew_at_base_' + str(threshold)
    if row[label_1] == 1:
        return 'offer'
    elif row[label_2] == 1:
        return 'base'
    else:
        return 'churn'


# In[497]:


#Combining them together

#Method takes in the churn_df, acceptance rate, and specified threshold

def process_table(churn_df, accept_offer_rate, threshold):
    
    #Get_Offer
    label_1 = 'get_offer_' + str(threshold)
    churn_df[label_1] = churn_df.apply(lambda x: get_offer(x['y_pred'], threshold), axis=1)
    
    #Accept Offer
    label_2 = 'accept_offer_' + str(threshold)
    churn_df[label_2] = churn_df[label_1].apply(accept_offer)
    
    #Renew at base
    label_3 = 'renew_at_base_' + str(threshold)
    churn_df[label_3] = churn_df.apply(lambda x: renew_at_base(x, threshold),axis=1)
    
    #get plan
    label_4 = 'plan_' + str(threshold)
    churn_df[label_4] = churn_df.apply(lambda x: get_plan(x,threshold),axis=1)
    
    return churn_df


# In[516]:


processed_churn_df = process_table(churn_df,accept_offer_rate,0.5)
processed_churn_df_30 = process_table(churn_df,accept_offer_rate,0.3)


# In[517]:


processed_churn_df_30


# In[519]:


processed_churn_df.groupby('plan_0.3').count()


# In[502]:


processed_churn_df.groupby('y_actual').count()


# In[ ]:


#Try with 0.7 threshold


# In[254]:


processed_churn_df_70 = process_table(churn_df,accept_offer_rate,0.7)


# In[256]:


processed_churn_df_70.groupby('plan_0.7').count()


# In[257]:


processed_churn_df_70.groupby('y_actual').count()


# Proforma calculations are continued on Excel

# # Calculating Customer CLV

# ## Get CAC for Each Customer

# In[295]:


#(this DF is from part 2)
ad_merged_filtered


# In[330]:


#Keep only necessary info for CLV/CAC Analysis
ad_merge_CAC = ad_merged_filtered[['year','month','attribution','CAC']]


# In[346]:


#Get mean CAC for imputation later
ad_merge_CAC['CAC'].mean()


# In[328]:


#This DF was from churn modelling section
df['account_creation_date_x'] = pd.to_datetime(df['account_creation_date_x'])

df['month'],df['year'] = df['account_creation_date_x'].dt.month, df['account_creation_date_x'].dt.year


# In[567]:


#Keep only necessary columns
df_CAC = df[['subid','month','year','attribution_technical','monthly_price','discount_price','join_fee']]
df_CAC.columns = ['subid','month','year','attribution','monthly_price','discount_price','join_fee']


# In[615]:


#Merge DF with CAC, on year, month and channel

df_subid_CAC = df_CAC.merge(ad_merge_CAC,how='left',on=['year','month','attribution'])
df_subid_CAC


# In[618]:


#Fill '0' for CAC for Organic Channels
organic_channels = ['organic','google_organic','bing_organic','pininterest_organic','facebook_organic']

def fill_organic(row):
    if row.attribution in organic_channels:
        return 0
    else:
        return row.CAC
    
df_subid_CAC['CAC_filled'] = df_subid_CAC.apply(fill_organic,axis=1)


# In[621]:


#For the rest, fill in the average CAC
df_subid_CAC['CAC_filled'] = df_subid_CAC['CAC_filled'].fillna(5.521)


# In[623]:


df_subid_CAC['join_fee'] = df_subid_CAC['join_fee'].fillna(0) #assume those with join fee NA value = 0 


# In[624]:


#Merge this with churn_df, which has the churn probabilities

df_CLV = churn_df.merge(df_subid_CAC,on='subid',how='left')


# In[626]:


df_CLV


# ## Getting CLV

# In[629]:


discount_rate = 0.03 #Assumption 0.1 annual discount rate, so for 4 months is 3%

def get_CLV(row):
    
    #timeframe here is one billing period, or 4 months
    clv = (row['discount_price']*4) * ((1+discount_rate)/((1+discount_rate) - (1- row['y_pred']))) 
    
    #clv add the revenue you already got from them, - CAC
    clv = clv + (row['discount_price'] * 4) + row['join_fee'] - row['CAC_filled']
    
    return clv

df_CLV['CLV_with_CAC'] = df_CLV.apply(get_CLV,axis=1)

#for CLV_without_CAC, remove row['CAC_filled'] from method and run again


# In[630]:


df_CLV


# In[635]:


#Export table of CLV to Tableau to plot
df_CLV.to_csv('distribution_of_CLV_v2.csv')

