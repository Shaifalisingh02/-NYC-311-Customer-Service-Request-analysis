#!/usr/bin/env python
# coding: utf-8

# ## Customer Service Requests Analysis.

# ### Analysis tasks  to performed.
# 
# 1.Import a 311 NYC service request.
# 
#  2.Read or convert the columns ‘Created Date’ and Closed Date’ to datetime datatype and create a new column 
#  ‘Request_Closing_Time’ as the time elapsed between request creation and request closing. (Hint: Explore the package/module datetime)
#  
# 3.Provide major insights/patterns that you can offer in a visual format (graphs or tables); at least 4 major conclusions that you can come up with after generic data mining.
# 
# 4.Order the complaint types based on the average ‘Request_Closing_Time’, grouping them for different locations.
# 
# 5.Perform a statistical test for the following:
# Please note: For the below statements you need to state the Null and Alternate and then provide a statistical test to accept or reject the Null Hypothesis along with the corresponding ‘p-value’.
# 
# Whether the average response time across complaint types is similar or not (overall)
# Are the type of complaint or service requested and location related?

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


#.Import a 311 NYC service request.
NYC=pd.read_csv('311_Service_Requests_from_2010_to_Present.csv')


# In[3]:


NYC.head()


# In[4]:


NYC.info()


# In[5]:


NYC.isnull().sum()


# ### *Descriptive Analysis*

# In[6]:


##Read or convert the columns ‘Created Date’ and Closed Date’ to datetime datatype.
NYC.describe()


# In[7]:


NYC.shape


# ####  Now we perform exploratory data analysis because all the  values given in the above does not provides us very clear insights.

# In[8]:


NYC['Created Date']=pd.to_datetime(NYC['Created Date'])
NYC['Closed Date']=pd.to_datetime(NYC['Closed Date'])


# In[9]:


#create a new column ‘Request_Closing_Time’ as the time elapsed between request creation and request closing.
NYC['Request_Closing_Time']=(NYC['Closed Date']-NYC['Created Date']).dt.total_seconds()
nyc1=NYC[NYC['Request_Closing_Time'].notnull()]
nyc_clean=nyc1[nyc1['Closed Date']>=nyc1['Created Date']]
nyc_clean['day of week']=nyc_clean['Created Date'].dt.dayofweek
nyc_clean['month']=nyc_clean['Created Date'].dt.day
nyc_clean['year']=nyc_clean['Created Date'].dt.year
nyc_clean=nyc_clean[nyc_clean.Borough!='Unspecified']


# In[10]:


nyc_clean.shape


# In[11]:


NYC['Agency'].unique()


# *All of our data belongs to a single agency NYPD i.e New York City Police Department.*

# In[12]:


#Univariate distribution plot for Request closing time.
sns.distplot(NYC['Request_Closing_Time'])
plt.show()


# In[13]:


NYC.head()


# ### Major Insights

# In[14]:


#Most frequent complaints
(nyc_clean['Complaint Type'].value_counts()).head().plot(kind='barh',figsize=(10,10),title='Most common complaints')


# In[15]:


#Least common complaints
(nyc_clean['Complaint Type'].value_counts()).tail().plot(kind='barh',figsize=(10,10),title='Most common complaints')


# In[16]:


nyc_clean.shape


# In[17]:


# complaints distribution across Borough
colors = ['blue','indigo','yellow','pink','magenta','cyan','violet']
nyc_clean['Borough'].value_counts().plot(kind='pie',autopct='%1.1f%%',explode=(0.15,0,0,0,0),startangle=45,shadow=False,colors=colors,figsize=(5,6))
plt.axis('equal')
plt.title("complaints distribution across Borough")
plt.tight_layout()
plt.show()


# ##### Maximum complaint requests were registered from Brooklyn. 

# In[18]:


brook=nyc_clean[nyc_clean['Borough']=='BROOKLYN']


# In[19]:


nyc_clean.shape


# In[20]:


brook.shape


# In[21]:


(brook['Complaint Type'].value_counts()).head(25).plot(kind='bar',figsize=(10,6),title = 'Most Frequent Complaints in Brooklyn')


# In[22]:


brook['Location Type'].value_counts().head().plot(kind='bar',figsize=(5,6),title='Location type vs Complaints type')


# In[23]:


nyc_clean[nyc_clean['Complaint Type']=='Blocked Driveway']['Descriptor'].value_counts()


# In[24]:


brook_blocked=brook[brook['Complaint Type'] == 'Blocked Driveway']


# In[25]:


brook_blocked.plot(kind='hexbin',x='Longitude',y='Latitude',gridsize=40,title='Blocked driveway issues concentration across Brooklyn\n',colormap='jet',mincnt=1,figsize=(10,6)).axis('equal')


# ### Request closing time in seconds,grouping them for different location.

# In[26]:


nyc_avg_res_time=nyc_clean.groupby(['City','Complaint Type']).Request_Closing_Time.mean()


# In[27]:


nyc_avg_res_time.head(18)


# In[28]:


nyc_avg_res_time=nyc_clean.groupby(['Complaint Type']).Request_Closing_Time.mean().sort_values(ascending=True)


# In[29]:


nyc_avg_res_time.head(15)


#  ### From the above, Null hypothesis:*avg response time across complaint type are not equal*
#  ### Alternate hypothesis: *avg response time across complaint type are equal*

# *Following complaints have resolution times which are very close. Disorderly Youth 12810.902098 ,Noise - Vehicle 12918.914430 One group can be formed for these complaints and one way Anova for this can be performed*

# In[30]:


nyc_des_youth=nyc_clean[nyc_clean['Complaint Type']=='Disorderly Youth']


# In[31]:


nyc_des_youth=nyc_des_youth.loc[:,['Request_Closing_Time']]


# In[32]:


nyc_des_youth.head()


# In[33]:


nyc_nos_veh=nyc_clean[nyc_clean['Complaint Type']=='Noise - Vehicle']
nyc_nos_veh=nyc_nos_veh.loc[:,['Request_Closing_Time']]
nyc_nos_veh.head()


# In[34]:


nyc_type_res=nyc_clean.loc[:,['Complaint Type','Request_Closing_Time']]
nyc_type_res.head()
nyc_type_res.columns


# In[37]:


import scipy.stats as stats
#stats f_oneway functions takes the groups as input and returns F and P-value
fvalue, pvalue = stats.f_oneway(nyc_des_youth, nyc_nos_veh)
pvalue


# Null hypothesis to be accepted for Disorderly Youth and Noise - Vehicle p-value close to 1

# In[39]:


nyc_post_adv=nyc_clean[nyc_clean['Complaint Type']=='Posting Advertisement']
nyc_post_adv=nyc_post_adv.loc[:,['Request_Closing_Time']]
nyc_post_adv.head()


# In[40]:


nyc_der=nyc_clean[nyc_clean['Complaint Type']=='Derelict Vehicle']
nyc_der=nyc_der.loc[:,['Request_Closing_Time']]
nyc_der.head()


# In[41]:


fvalue, pvalue = stats.f_oneway(nyc_post_adv, nyc_der)
pvalue


# Null hypothesis for Posting Advertisement and Derelict Vehicle to be rejected p-value < 0.05

# In[ ]:




