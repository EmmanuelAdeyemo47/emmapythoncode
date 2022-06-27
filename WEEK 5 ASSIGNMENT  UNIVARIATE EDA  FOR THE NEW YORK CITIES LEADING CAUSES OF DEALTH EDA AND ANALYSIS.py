#!/usr/bin/env python
# coding: utf-8

# ### Import necessary dependencies

# In[71]:


import scipy.stats as st
import pandas as pd
import re
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[ ]:





# In[ ]:





# ### Read and Load the Dataset
# Importing the New York City Leading Causes of Death dataset

# In[2]:


csv = r'C:\Users\User\Documents\North Central University PHD\TIM8501 COURSE 2\ny leading cd.csv'
df = pd.read_csv(csv, sep=',')

# Viewing the Dataset.

df.tail(10)
df.head(5)


# ### Now Let perform EDA on this Dataset following these steps:
# ### Step 1: Determine the variables and data types of the dataset.¶

# In[3]:


#LETS SEE THE variables Information

df.info()


# ### Output
# From the output, the New York City Leading Causes of Death dataframe 
# has a total of 7 columns and a total of 1,272 rows. 1 of the columns is interger of type and 6 were of object type.
# 
# To be able to compute and plot graph easily we need to convert some object type column to float type column.

# ## Step2: Determine the Shape of the dataset

# In[4]:


df.shape


# ### Output
# From the output, The dataset contains 1,272 rows and 7 columns. It will be nice to see 
# the distribution of the 3 quantitative coulmns (Death, Death Rate and Age Adjusted Date Rate) and the frequency counts of the 
# other 4 qualitative variables (year, Leadingcause, Sex, and Race Ethnicity) of Leading Causes of Death in New York City.
# 

# In[5]:


#plt.figure(figsize=(16,3))

#plt.subplot(1,2,1)
#sns.boxplot(df['Deaths'])

#plt.subplot(1,2,2)
#sns.distplot(df['Death Rate'])

#plt.subplot(1,2,3)
#sns.distplot(df['Age Adjusted Death Rate'])

#plt.show()
#sns.distplot(df['Deaths'], hist=True, kde=True, color='g')

#plt.show()


# ### Output we cant plot this now because of missing values

# ### Step 3: Check for Missing Data and Anomalies

# In[6]:


pd.isnull(df).sum().sum()


# In[7]:


# Lets check the column with the missing data.
# which column has missing data?

df.loc[:, df.isnull().any()].columns


# In[8]:


# Checking the columns with its number of missing data

pd.isnull(df).sum()


# ### Output
# After runing the codes, we find out that  there are missing data points in in dataset. we went further to see the colunms with missing data and it was discovered that Death Rate has 67 missing datapoints, Age Adjusted Death Rate has 67 datapoints and the total missing data point in the whole dataset adds upto be 134.

# ### Let Remedy the Missing Data in the data set

# In[9]:


# One way to remedy the missing data is to use drop(na) since our dataset is large eneough

df.dropna()


# ### Output
# There are no more missing data in the New York City Leading Causes of Death because
# all the missing values count are now zero but there are enteries with a "dot" value so we need to remove all 
# these dots from our dataframe using the repalce() and dropna() method.

# In[10]:


# Set the new data frame to be equal new_df

new_df= df.dropna()


# In[11]:


new_df


# In[12]:


# Replacing  the dots value with nan and droping the nan missing data

df1 = new_df.replace('.', np.nan).dropna()


# In[13]:


# Viewing the dataset

df1.head(5)


# In[14]:


# Veiwing the shape ofthe data
df1.shape

#Out

#As we can see the shpe of the data has been reduced from (1272, 7) to (819, 7) and all the dot values have been removed.


# In[15]:


# Checking to confirm if there are still any missing data

pd.isnull(df1).sum()


# ### Output 
# 
# There are no more missing data in the New York City Leading Causes of Death because all the missing values count are now zero
# 

# In[16]:


df1.info()


# ### Convert Object to float type in these (Deaths,DeathRate, AgeAdjustedDeathRate)
# ### variables for easy ploting and computation.

# In[17]:


# Convert Object to float type in these (Deaths,DeathRate, AgeAdjustedDeathRate)variables for easy ploting and computation

df1['Deaths'] = df1.Deaths.astype(float)


# In[18]:



df1['AgeAdjustedDeathRate'] = df1.AgeAdjustedDeathRate.astype(float)


# In[19]:



df1['DeathRate'] = df1.DeathRate.astype(float)


# In[20]:


# "Year" will be Treated as Object

df1['Year'] = df1.Year.astype(object)


# In[21]:


# Now let check the variables datatype again

df1.info()

#output:
#As we can see all the desired variables have changed as desired.


# In[22]:


# Let see the distribution of the three float column


plt.figure(figsize=(16,3))

plt.subplot(1,2,1)
sns.distplot(df1['Deaths'])

plt.subplot(1,2,2)
sns.distplot(df1['DeathRate'])

plt.show()

#sns.distplot(df['Deaths'], hist=True, kde=True, color='g')

#plt.show()


# In[23]:


plt.figure(figsize=(16,3))

plt.subplot(1,2,1)
sns.distplot(df1['AgeAdjustedDeathRate'])

plt.show()


# ### output:
# The New York City Leading Causes of Death has a skewed distribution.

# In[24]:


# Ploting individual histogram for the three variables

fig = df1.hist(bins=15, color='red', edgecolor='darkmagenta', linewidth=1.0, xlabelsize=10, ylabelsize=10, xrot=45, yrot=0, figsize=(5,5), grid=False)

plt.tight_layout(rect=(0, 0, 1.5, 1.5)) 


#Output:

# Since the data values seem to pile up into a single "mound" that means the distribution

# of The New York City Leading Causes of Death for Death, DeathRate and AgeAdjustedDeathRtae variables are  unimodal.


# ### Exploring the Qualitative Column: Leading Cause of Death, Sex and Race Ethnicity 
# ### through frequency count.

# In[25]:


df1.groupby(['Leading Cause']).count()


# In[26]:


#Creating Frequency Table for Leading Cause of Deaths in New  york city.

freq_table= df1.groupby(['Leading Cause']).size().reset_index(name='Count').rename(columns={'Leading Cause':'Leading Cause'})
freq_table


# In[27]:


#Lets Plot a Bar-Chart for the Leading Cause of Death

plt.bar(freq_table['Leading Cause'],freq_table['Count'])
plt.show()


# ### output:
# 
# From the output, According to the dataset,there were 24 leading causes of death in New York city.
# The highest leading cause of death in New York city are: Diabetes Mellitus (E10-E14), Influenza (Flu) and Pneumonia (J09-J18), Malignant Neoplasms (Cancer: C00-C97) and All other causes with 74 Death count each. The second Leading causes of death in New york city is Cerebrovascular Disease (Stroke: I60-I69) with 73 Death counts, follow by Essential Hypertension and Renal Diseases with a death count of 64 while the least causes of Death in New york City are Assault (Homicide: U01-U02, Y87.1, X85-Y09),Congenital Malformations, Deformations, Mental and Behavioral Disorders and Viral Hepatitis (B15-B19) with a deathcount of 1 each. 			
# 
# 

# In[ ]:





# In[28]:


df1.groupby(['Sex']).count()


# In[29]:


#Creating Frequency Table for the  leading Cause of  Death in New York City by SEX

freq_table= df1.groupby(['Sex']).size().reset_index(name='Count').rename(columns={'Sex':'Sex'})
freq_table


# In[30]:


#Lets Plot a Bar-Chart for the  leading Cause of  Death in New York City by Sex

plt.bar(freq_table['Sex'],freq_table['Count'])
plt.show()


# ### Ouput
#   From the output, the Leading Cuase of Death in New york city shows that male and female have the same number 
#   of death counts of 354.

# In[31]:


df1.groupby(['RaceEthnicity']).count()


# In[32]:


#Creating Frequency Table for the  leading Cause of  Death in New York City by Race_Ethnicity

freq_table= df1.groupby(['RaceEthnicity']).size().reset_index(name='Count').rename(columns={'RaceEthnicity':'RaceEthnicity'})
freq_table


# In[33]:


#Lets Plot a Bar-Chart for the  leading Cause of  Death in New York City by Race_Ethnicity

plt.bar(freq_table['RaceEthnicity'],freq_table['Count'])
plt.show()


# ### output:
# 
# from the bar-chart above we saw that highes number of Death by Ethnicity is recorded among the people of  
# Asian and Pacific Islander with a total death count of 199 and same for the Hispanic ethnicity group too. 178 Deaths recorded
# among Black Non-Hispanic group and 176 deaths among white-non Hispanic, 22 deaths were recorded among Black Non- Hispanic
# group and 23 deaths recorded among Other ethnicity group.

# In[34]:


df1.groupby(['Year']).count()


# In[35]:


#Creating Frequency Table for yealy record of Death in New York City

freq_table= df1.groupby(['Year']).size().reset_index(name='Count').rename(columns={'Year':'Year'})
freq_table


# ### output
# 
# As we can see the Highest number of Deaths were recorded in New york city in the year 2019 with 111 number of Death, 
# follow by 2008, 2009 with 89 total deaths, 2010 through 2014 total number of Death recorded for each ofthese years
# is 88 each. 

# In[36]:


#Lets Plot a Bar-Chart for the yearly record of Death in New York City.

plt.bar(freq_table['Year'],freq_table['Count'])
plt.show()


# ### Step 4: Identify significant correlations and relationships among variables¶
# ### in the dataset using heatmap plot

# In[37]:


#sns.heatmap(new_df.corr())

fig, (ax) = plt.subplots(1, 1, figsize=(7,4))

sns.heatmap(df1.corr(), 
                ax=ax, 
                 cmap="bwr", 
                 annot=True, 
                 fmt='.2f', 
                 linewidths=.05)

fig.subplots_adjust(top=0.83)
fig.suptitle('Correlation among Deaths, Death rate and AgeAdjustedDeathRate ', fontsize=14, fontweight='bold')


# ### Output 
# 
# From the correlation matrix plot above, we can see that all the three variables are positively
# correlated and and the scatter plot below shows relationships among these variables. This correlation can be investigated as a further study. we will look into this correlated relationship later.  
# 

# In[38]:


# Scatter plot showing relationship for Deaths, Deathrate and AgeAdjustedDeathRate variables.

sns.pairplot(df1)


# ### Step 5: Determine the extent of Outliers
# Let SEE THE OUTLIERS In the New York Leading Cause of Death Quantitative Variables

# In[39]:


# Boxplot to detect outliers

sns.boxplot(df1['Deaths'])


# In[40]:


# Lets treat our outliers with log10 transformation.
 # Cretaing dataframe for only Deaths variable
    
death_array = np.log10(df1['Deaths'])

print(death_array)
    
    
#df1.describe()


# In[41]:


# NOW THAT WE HAVE TRANSFORMED THE DEATH DATA, LETS US NOW PLOT THE BOXPLOT AGAIN

sns.boxplot(death_array)


# In[42]:



sns.distplot(death_array, hist=True, kde=True, color='r')


# ### OUTPUT. 
# Log transform does not seem to resolve the problem of outlies. From the box plot we can still sport outliers.

# ### Since we have outliers in the dataset, its important to get rid of these outliers before
# computing the statistics. We will use IQR based filtering to remedy the outliers since our 
# data distribution is skewed.

# In[43]:


# Finding the IQR and upper and Lower Limits.

percentile25 = df1['Deaths'].quantile(0.25)

percentile75 = df1['Deaths'].quantile(0.75)

# Finding iqr

iqr = percentile75 - percentile25
print(iqr)

## Finding upper and lower limit

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

print('upper limit:', upper_limit)
print('lower limit:', lower_limit)


# ### Lets Cap the Dataset

# In[44]:


# Capping the clean_df dataset
    
    
clean_df_cap = df1.copy()
clean_df_cap['Deaths'] = np.where(
    clean_df_cap['Deaths'] > upper_limit,
    upper_limit,
    np.where(
        clean_df_cap['Deaths'] < lower_limit,
        lower_limit,
        clean_df_cap['Deaths']
    )
)


# In[45]:


# Let Compare the plots after capping to see if we have completely taking care of all the outliers

plt.figure(figsize=(10,5))
plt.subplot(2,2,1)

sns.distplot(df1['Deaths'])
plt.subplot(2,2,2)
sns.boxplot(df1['Deaths'])
plt.subplot(2,2,3)

sns.distplot(df1['Deaths'])
plt.subplot(2,2,4)
sns.boxplot(clean_df_cap['Deaths'])
plt.show()


# In[46]:


#output: 

# From the above figure, we have completely removed outliers in our dataset(clean_df_cap) through capping.


# In[47]:


print(clean_df_cap.shape)


# ### Now let compute the basic statistics of the the  Leading cause of Death in NewYork city  dataset

# In[48]:


clean_df_cap.describe()


# ### CHI-SQAURE TEST FOR OUR HYPOTHESIS TESTING

# ### Research Question1: What are the most yearly leading causes of death in New York Cities? 

# In[49]:


#Research Question1: What are the most yearly leading causes of death in New York Cities? 
#Through frequency table and bar chart 


# ### Research Question 2: Is there a Relationship between Sex and leading cause of death in New York city?

# In[50]:


# Research Question 2: Is there a Relationship between Sex and leading cause of death in New York city?

#Ho: There is no significant relationship between Sex and Leading cause of death in NewYOrk City
# H1: There is significant Relationship.

#Test Statistics: chi-sqare-calculated 

# Decision Rule: Reject Ho: If P-value < Alpha.    Alpha= 0.05


# In[51]:


# We set the explored and clean dataset as dataset

dataset = clean_df_cap


# In[52]:


# This Gives the cross tabulation of male and female Leading cuase of death.

dataset2= pd.crosstab(dataset['Sex'], dataset['Leading Cause'])
#print(dataset)


# In[53]:


# observed values

observed_values = dataset2.values
print("observed Values :-\n",observed_values)


# In[54]:


# This is used to calculate the Chi-square statistic value

val=st.chi2_contingency(dataset2)


# In[55]:


#Remove the hashtag behind val to display the chi-square vales, p-value and degree of freedom

#val


# ### Output
# Chi-sqaure = 242.17 , P-value= 0.0000, df= 72, alpha =0.05
# 
# Decision Rule = Reject Ho: if p-value is < alpha
# 
# Decision: We Reject Ho and Acccept H1:
# 
# Conclusion: There is significant relationship between Sex and Leading cause of death in New York City.
# 

#  ### Research Question 3: Is there a Relationship between Race Ethnicities and leading cause of death in New York cities?

# In[56]:


#Ho: There is no significant relationship between  Race Ethnicities  and Leading cause of death in NewYOrk City
# H1: There is significant Relationship.

#Test Statistics: chi-sqare-calculated 

# Decision Rule: Reject Ho: If P-value < Alpha.    Alpha= 0.05


# In[57]:


dataset3= pd.crosstab(dataset['RaceEthnicity'], dataset['Leading Cause'])
#print(dataset3)


# In[58]:


# observed values

observed_values = dataset3.values
#print("observed Values :-\n",observed_values)

val=st.chi2_contingency(dataset3)

#Remove the hashtag behind val to display the chi-square vales, p-value and degree of freedom

#val


# ### Output
# Chi-sqaure = 342.02 , P-value= 0.0000, df= 144, alpha =0.05
# 
# Decision Rule = Reject Ho: if p-value is < alpha
# 
# Decision: We Reject Ho and Acccept H1:
# 
# Conclusion: There is significant relationship between RaceEthnicity and Leading cause of death in New York City.

# ### Research Question 4: Is there a Relationship between Age Adjusted Death Rate 
# ### and leading cause of death in New York cities?
# 

# In[59]:


#Ho: There is no significant relationship between  Age Adjusted Death  and Leading cause of death in NewYork City
# H1: There is significant Relationship.

#Test Statistics: chi-sqare-calculated 

# Decision Rule: Reject Ho: If P-value < Alpha.    Alpha= 0.05


# In[60]:


dataset4= pd.crosstab(dataset['AgeAdjustedDeathRate'], dataset['Leading Cause'])
#print(dataset4)

# observed values

observed_values = dataset4.values
#print("observed Values :-\n",observed_values)

val=st.chi2_contingency(dataset4)

#Remove the hashtag behind val to display the chi-square vales, p-value and degree of freedom

#val


# ### Output
# Chi-sqaure = 14529.78 , P-value= 0.0000, df= 12864, alpha =0.05
# 
# Decision Rule = Reject Ho: if p-value is < alpha
# 
# Decision: We Reject Ho and Acccept H1:
# 
# Conclusion: There is significant relationship between AgeAdjustedDeathRate and Leading cause of death in New York City.

# ### Research Question 5: Is there a Relationship between Sex and number of Death in New York cities?

# In[61]:


#Ho: There is no significant relationship between number Sex and  death in NewYork City
# H1: There is significant Relationship.

#Test Statistics: chi-sqare-calculated 

# Decision Rule: Reject Ho: If P-value < Alpha.    Alpha= 0.05


# In[62]:


dataset5= pd.crosstab(dataset['Sex'], dataset['Deaths'])
#print(dataset5)

# observed values

observed_values = dataset5.values
#print("observed Values :-\n",observed_values)

val=st.chi2_contingency(dataset5)

#Remove the hashtag behind val to display the chi-square vales, p-value and degree of freedom

#val


#  ### Output
# Chi-sqaure = 1094.53 , P-value= 0.07, df= 1026, alpha =0.05
# 
# Decision Rule = Reject Ho: if p-value is < alpha
# 
# Decision: We Accept  Ho 
# 
# Conclusion: There is no significant relationship Sex and number of  death in New York City.

# ### Research Question 6: Is there a Relationship between Sex and Age Adjusted Death Rate in New York cities?

# In[63]:


#Ho: There is no significant relationship between number Sex and  Age Adjusted death rate in NewYork City
# H1: There is significant Relationship.

#Test Statistics: chi-square-calculated 

# Decision Rule: Reject Ho: If P-value < Alpha.    Alpha= 0.05


# In[64]:


dataset6= pd.crosstab(dataset['Sex'], dataset['AgeAdjustedDeathRate'])
#print(dataset6)

# observed values

observed_values = dataset6.values
#print("observed Values :-\n",observed_values)

val=st.chi2_contingency(dataset6)

#Remove the hashtag behind val to display the chi-square vales, p-value and degree of freedom

#val


# ### Output
# Chi-sqaure = 2200.04 , P-value= 0.00, df= 1608, alpha =0.05
# 
# Decision Rule = Reject Ho: if p-value is < alpha
# 
# Decision: We RejECT Ho and accept H1
# 
# Conclusion: There is significant relationship between Sex and AgeAdjustedDeathRate in New York City.
# 

# ### Research Question 7: Is there significant difference in the mean of Deaths, Death rate and and Age Adjusted Death Rate in the New York City Leading Causes of Death?
# 
# 

# In[65]:


#Ho: There is no significant difference in the mean of Deaths, Death rate and and Age Adjusted Death Rate 
#in the New York City Leading Causes of Death?

# H1: There is significant Relationship.

#Test Statistics: One way Anova

# Decision Rule: Reject Ho: If P-value < Alpha.    Alpha= 0.05


# In[66]:


# Extract individual groups for ANALYSIS OF VARIANCE.

Deaths = dataset['Deaths']

DeathRate = dataset['DeathRate']

AgeAdjustedDeathRate = dataset['AgeAdjustedDeathRate']


# In[67]:


# Perform the Anova

st.f_oneway(Deaths, DeathRate, AgeAdjustedDeathRate )


# ### Output
# F= 559.48 , P-value= 0.00, alpha =0.05
# 
# Decision Rule = Reject Ho: if p-value is < alpha
# 
# Decision: We RejECT Ho and accept H1
# 
# Conclusion: There is significant difference in the means of Deaths, Death rate and and Age Adjusted Death Rate 
# in the New York City Leading Causes of Death.

# In[68]:


# Create the data
x = dataset['Year']
y = dataset['Deaths']

# Let's plot the data
plt.plot(x, y,label='Deaths', color='blue')

# Create the data

x = dataset['Year']
y = dataset['DeathRate']

# Plot the data
plt.plot(x, y, label='DeathRate', color='red')

# Add X Label on X-axis
plt.xlabel("Year")

# Add X Label on X-axis
plt.ylabel("Death and Death rate")

# Append the title to graph
plt.title("New York Leading Causes of Death")

# Add legend to graph
plt.legend()

# Display the plot
plt.show()


# ###  RUNING MULTIPLE LINEAR REGRESSION

# In[72]:


# Our clean dataset

dataset = clean_df_cap


# In[73]:


dataset.Deaths.median()


# In[ ]:




