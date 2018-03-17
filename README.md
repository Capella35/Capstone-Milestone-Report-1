# Capstone-Milestone-Report-1/ Related Codes
import pandas as pd
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
%matplotlib inline
import scipy.stats as st
import numpy as np

# 

df = pd.read_csv('/Users/emrahceyhan/Desktop/SpringBoardArchieve/CapstoneProject/Capstone1-HR-Employee-Attrition.csv')
#Explore the data
df.info()
df.head()
df.shape()
#Looking for NaN
df.isnull().any()
#Check the unique columns in the data to find which are categorical
nunique=df.nunique()
nunique= nunique.sort_values()
nunique
#Delete columns having the value of '1 or any same values for all observation'. 
cols = ["Over18", "StandardHours", "EmployeeCount"]
for i in cols:
    del df[i]
#Relabeled of some colums
df['Education_r'] = pd.cut(df['Education'], 5, labels=['Below College','College','Bachelor','Master','Doctor'])
df['EnvironmentSatisfaction_r']= pd.cut(df['EnvironmentSatisfaction'], 4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
df['JobInvolvement_r']= pd.cut(df['JobInvolvement'], 4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
df['JobSatisfaction_r']= pd.cut(df['JobSatisfaction'], 4, labels=['Low', 'medium', 'High', 'VeryHigh'])
df['PerformanceRating_r']= pd.cut(df['PerformanceRating'], 4, labels= ['Low', 'Good', 'Outstanding', 'Excellent'])
df['RelationshipSatisfaction_r']= pd.cut(df['RelationshipSatisfaction'], 4, labels= ['Low', 'Medium', 'Outstanding','Excellent'])
df['WorkLifeBalance_r']= pd.cut(df['WorkLifeBalance'], 4, labels=['Bad', 'Good', 'Better', 'Best'])
#Use the pandas get_dummies() function to create dummy variables from the df DataFrame.
df['Attrition_r']= pd.get_dummies(df.Attrition, drop_first = True)
df.head()

#Recode Attrition column to get 0=No, 1=Yes
df['Attrition_r']= pd.get_dummies(df.Attrition, drop_first = True)
df.Attrition_r.head()

#Create a new data set called 'df_yes' by filtering the data set 'df' where Attrition='Yes' 
df_yes=df[df.Attrition == 'Yes']
print(df_yes.head())
df_yes.shape

#Histogram of YearsAtCompany Vs Department
sns.barplot(x = 'Department', y = 'YearsAtCompany', data = df_yes)
plt.show()
fig,ax = plt.subplots(4,3, figsize=(12,13)) 

#Display multiple distribution plots
#'ax' has references to all the four axes
plt.suptitle("Distribution of various individual factors", fontsize=20)
sns.distplot(df_yes['TotalWorkingYears'], ax = ax[0,0]) 
sns.distplot(df_yes['YearsAtCompany'], ax = ax[0,1]) 
sns.distplot(df_yes['DistanceFromHome'], ax = ax[0,2]) 
sns.distplot(df_yes['YearsInCurrentRole'], ax = ax[1,0]) 
sns.distplot(df_yes['YearsWithCurrManager'], ax = ax[1,1]) 
sns.distplot(df_yes['YearsSinceLastPromotion'], ax = ax[1,2]) 
sns.distplot(df_yes['PercentSalaryHike'], ax = ax[2,0]) 
sns.distplot(df_yes['YearsSinceLastPromotion'], ax = ax[2,1]) 
sns.distplot(df_yes['TrainingTimesLastYear'], ax = ax[2,2]) 
sns.distplot(df_yes['Age'], ax=ax[3,0])
sns.distplot(df_yes['PerformanceRating'], ax=ax[3,1])
sns.distplot(df_yes['HourlyRate'], ax=ax[3,2])
plt.show()

#The goal is to identify some features that link to where Attrition='Yes'
from numpy import median
fig,ax = plt.subplots(2,3, figsize=(20,20))               # 'ax' has references to all the four axes
plt.suptitle("Distribution of various factors with other factors", fontsize=20)
sns.barplot(df_yes['Department'],df_yes['MonthlyIncome'],hue = df_yes['OverTime'], estimator=median, ax = ax[0,0]); 
sns.barplot(df_yes['BusinessTravel'],df_yes['YearsAtCompany'],hue = df['OverTime'], estimator=median, ax = ax[0,1]); 
sns.barplot(df_yes['JobLevel'],df_yes['MonthlyIncome'],hue = df_yes['Department'], estimator=median, ax = ax[0,2]); 
sns.barplot(df_yes['Department'],df_yes['YearsAtCompany'],hue = df_yes['JobInvolvement'],estimator=median, ax = ax[1,0]); 
sns.barplot(df_yes['Department'],df_yes['MonthlyIncome'],hue = df_yes['Gender'], estimator=median, ax = ax[1,1]); 
sns.barplot(df_yes['JobLevel'],df_yes['DistanceFromHome'],hue = df_yes['Department'], estimator=median, ax = ax[1,2]);

plt.show()

#Compare average and median monthly rate of men and women who were left the job
avg_male_rate = np.mean(df_yes.MonthlyRate[df.Gender == 'Male']) 
avg_female_rate = np.mean(df_yes.MonthlyRate[df.Gender == 'Female'])
med_male_rate = np.median(df_yes.MonthlyRate[df.Gender == 'Male']) 
med_female_rate = np.median(df_yes.MonthlyRate[df.Gender == 'Female'])
plt.bar([1,4],[avg_male_rate, med_male_rate]) 
plt.bar([2,5],[avg_female_rate, med_female_rate], color = 'y') 
plt.xticks([2,5],['Average Monthly Rate','Median Monthly Rate']) 
plt.legend(['Males','Females'], loc = 2)

print(avg_female_rate/avg_male_rate) 

#Build factor plots for categorical values towards one numerical values
sns.factorplot(x =   'Gender',     # Categorical
               y =   'DistanceFromHome',      # Continuous
               hue = 'MaritalStatus',    # Categorical
               col = 'JobRole',
               col_wrap=2,           # Wrap facet after two axes
               kind = 'box',
               data = df_yes)
plt.show()

#Plot a correlation map for all numeric variables
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#Apply two sample t-test for age & attrition, distancefromhome & attrition,  TotalWorkingYears & attrition and
#YearsSinceLastPromotion & Attrition. 
two_sample = st.ttest_ind(df_no['Age'], 
                          df_yes['Age'])
print('The t-statistic is %.3f and the p-value is %.8f.' % two_sample)

two_sample = st.ttest_ind(df_no['DistanceFromHome'], 
                          df_yes['DistanceFromHome'])
print('The t-statistic is %.3f and the p-value is %.8f.' % two_sample)

two_sample = st.ttest_ind(df_no['TotalWorkingYears'], 
                          df_yes['TotalWorkingYears'])
print('The t-statistic is %.3f and the p-value is %.8f.' % two_sample)

two_sample = st.ttest_ind(df_no['YearsSinceLastPromotion'], 
                          df_yes['YearsSinceLastPromotion'])
print('The t-statistic is %.3f and the p-value is %.8f.' % two_sample)


