#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[64]:


df = pd.read_csv('StudentsPerformance.csv')


# In[65]:


df


# In[66]:


df.dtypes


# In[67]:


df.info()


# In[68]:


df.notnull()


# In[69]:


df.value_counts()


# In[70]:


df.isnull()


# In[71]:


df.iloc[[3,4]]


# In[72]:


df.iloc[[0,1],[3,4]]


# In[73]:


df.columns


# In[74]:


df.Math_score.mode


# In[75]:


df.Math_score.mean


# In[76]:


df1=df


# In[77]:


df1.fillna(0)


# In[78]:


df1


# In[79]:


df["Math_score"].fillna(df["Math_score"].mean(),inplace = True)


# In[80]:


df


# In[81]:


df["Reading_score"].fillna(df["Reading_score"].mean(),inplace = True)


# In[82]:


df


# In[83]:


df["Writing_score"].fillna(df["Writing_score"].mean(),inplace = True)


# In[84]:


df


# In[85]:


col =['Math_score','Reading_score','Writing_score','Placement_score']


# In[86]:


df.boxplot(col)


# In[87]:


z = stats.zscore(df['Math_score'])


# In[88]:


z


# In[89]:


thershold = 2


# In[90]:


outlier = np.where(z>thershold)


# In[91]:


outlier


# In[92]:


Q1 = df.Math_score.quantile(0.25)


# In[93]:


Q3 = df.Math_score.quantile(0.75)


# In[94]:


Q1


# In[95]:


Q3


# In[96]:


IQR = Q3 - Q1


# In[97]:


IQR


# In[98]:


lower_limit = Q1 - 1.5 * IQR


# In[99]:


lower_limit


# In[100]:


upper_limit = Q3 + 1.5 * IQR


# In[101]:


upper_limit


# In[102]:


outlier = df["Math_score"][(df["Math_score"] > upper_limit) | (df["Math_score"] < lower_limit)] 


# In[103]:


outlier


# In[104]:


mean1 = df['Math_score'].mean()


# In[105]:


mean1


# In[106]:


df["Math_score"] = np.where((df['Math_score']>upper_limit), mean1, df["Math_score"])


# In[107]:


df["Math_score"]


# In[108]:


Q1 = df.Reading_score.quantile(0.25)


# In[109]:


Q1


# In[110]:


Q3 = df.Reading_score.quantile(0.75)


# In[111]:


Q3


# In[112]:


IQR = Q3 - Q1


# In[113]:


IQR


# In[114]:


lower_limit = Q1 - 1.5 * IQR


# In[115]:


lower_limit


# In[116]:


upper_limit = Q3 + 1.5 * IQR


# In[117]:


upper_limit


# In[118]:


mean1 = df['Reading_score'].mean()


# In[119]:


mean1


# In[120]:


df["Reading_score"] = np.where((df['Reading_score']>upper_limit), mean1, df["Reading_score"])


# In[121]:


df["Reading_score"]


# In[122]:


df.isnull()


# In[123]:


print("skewness : ",df["Math_score"].skew())
print("kurtosis : ",df["Math_score"].kurtosis())


# In[125]:


plt.hist(df["Math_score"], bins=10, edgecolor = 'black')
plt.title("Distribution of Math_score")
plt.xlabel("Math_score")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# In[135]:


sns.histplot(df["Math_score"].dropna(), kde = True, stat = "count", color = "red", bins = 10)
plt.title("Histogram with kde")
plt.xlabel("Math_score")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


# In[138]:


df["Math_score_log"] = np.log1p(df["Math_score"])
df["Math_score_log"]


# In[139]:


plt.hist(df["Math_score_log"], bins=10, edgecolor = 'black')
plt.title("Distribution of Math_score")
plt.xlabel("Math_score_log")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# In[140]:


sns.histplot(df["Math_score_log"].dropna(), kde = True, stat = "count", color = "red", bins = 10)
plt.title("Histogram with kde")
plt.xlabel("Math_score")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


# In[142]:


min_math_score = df['Math_score'].min()


# In[143]:


min_math_score


# In[145]:


max_math_score = df['Math_score'].max()


# In[146]:


max_math_score


# In[147]:


plt.hist(df["Reading_score"], bins=10, edgecolor = 'black')
plt.title("Distribution of Reading_score")
plt.xlabel("Math_score_log")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# In[148]:


sns.histplot(df["Reading_score"].dropna(), kde = True, stat = "count", color = "red", bins = 10)
plt.title("Histogram with kde")
plt.xlabel("Reading_score")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


# In[149]:


df["Reading_score_log"] = np.log1p(df["Reading_score"])
df["Reading_score_log"]


# In[151]:


sns.histplot(df["Reading_score_log"].dropna(), kde = True, stat = "count", color = "red", bins = 10)
plt.title("Histogram with kde")
plt.xlabel("Reading_score")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


# In[156]:


sns.histplot(df["Club_Join_Date"].dropna(), kde = True, stat = "count", color = "red", bins = 10)
plt.title("Histogram with kde")
plt.xlabel("Club_Join_Date")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


# In[157]:


sns.histplot(df["Writing_score"].dropna(), kde = True, stat = "count", color = "red", bins = 10)
plt.title("Histogram with kde")
plt.xlabel("Writing_score")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


# In[158]:


df["Writing_score_log"] = np.log1p(df["Writing_score"])
df["Writing_score_log"]


# In[159]:


sns.histplot(df["Writing_score_log"].dropna(), kde = True, stat = "count", color = "red", bins = 10)
plt.title("Histogram with kde")
plt.xlabel("Writing_score")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


# In[160]:


df["Club_Join_Date_log"] = np.log1p(df["Club_Join_Date"])
df["Club_Join_Date_log"]


# In[161]:


sns.histplot(df["Club_Join_Date_log"].dropna(), kde = True, stat = "count", color = "red", bins = 10)
plt.title("Histogram with kde")
plt.xlabel("Club_Join_Date")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


# In[173]:


plt.hist(df["Writing_score"], bins=10, color = "Green", edgecolor = 'black')
plt.title("Distribution of Writing_score")
plt.xlabel("Writing_score")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


# In[169]:


plt.hist(df["Club_Join_Date"], bins=10, color = "purple", edgecolor = 'black')
plt.title("Distribution of Club_Join_Date")
plt.xlabel("Club_Join_Date")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


# In[186]:


df['Club_Join_Date'].value_counts().plot(kind='bar')


# In[187]:


df = df.loc[df["Math_score"] >= 1 ]


# In[195]:


fare = df['Math_score']
fare = np.log(fare)
graph=sns.distplot(fare,label="Skewness: %.2f"%(fare.skew()))
graph.legend()


# In[196]:


print("skewness : ",df["Math_score"].skew())
print("kurtosis : ",df["Math_score"].kurtosis())


# In[219]:


df['Club_Join_Date'].value_counts().plot(kind='pie')


# In[224]:


df['Math_score'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(10, 10), title='Math_score')


# In[223]:


df['Club_Join_Date'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(10, 10), title='Club Join Date Distribution')


# In[ ]:




