
# coding: utf-8

# # Assignment 4
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# This assignment requires that you to find **at least** two datasets on the web which are related, and that you visualize these datasets to answer a question with the broad topic of **economic activity or measures** (see below) for the region of **Ann Arbor, Michigan, United States**, or **United States** more broadly.
# 
# You can merge these datasets with data from different regions if you like! For instance, you might want to compare **Ann Arbor, Michigan, United States** to Ann Arbor, USA. In that case at least one source file must be about **Ann Arbor, Michigan, United States**.
# 
# You are welcome to choose datasets at your discretion, but keep in mind **they will be shared with your peers**, so choose appropriate datasets. Sensitive, confidential, illicit, and proprietary materials are not good choices for datasets for this assignment. You are welcome to upload datasets of your own as well, and link to them using a third party repository such as github, bitbucket, pastebin, etc. Please be aware of the Coursera terms of service with respect to intellectual property.
# 
# Also, you are welcome to preserve data in its original language, but for the purposes of grading you should provide english translations. You are welcome to provide multiple visuals in different languages if you would like!
# 
# As this assignment is for the whole course, you must incorporate principles discussed in the first week, such as having as high data-ink ratio (Tufte) and aligning with Cairo’s principles of truth, beauty, function, and insight.
# 
# Here are the assignment instructions:
# 
#  * State the region and the domain category that your data sets are about (e.g., **Ann Arbor, Michigan, United States** and **economic activity or measures**).
#  * You must state a question about the domain category and region that you identified as being interesting.
#  * You must provide at least two links to available datasets. These could be links to files such as CSV or Excel files, or links to websites which might have data in tabular form, such as Wikipedia pages.
#  * You must upload an image which addresses the research question you stated. In addition to addressing the question, this visual should follow Cairo's principles of truthfulness, functionality, beauty, and insightfulness.
#  * You must contribute a short (1-2 paragraph) written justification of how your visualization addresses your stated research question.
# 
# What do we mean by **economic activity or measures**?  For this category you might look at the inputs or outputs to the given economy, or major changes in the economy compared to other regions.
# 
# ## Tips
# * Wikipedia is an excellent source of data, and I strongly encourage you to explore it for new data sources.
# * Many governments run open data initiatives at the city, region, and country levels, and these are wonderful resources for localized data sources.
# * Several international agencies, such as the [United Nations](http://data.un.org/), the [World Bank](http://data.worldbank.org/), the [Global Open Data Index](http://index.okfn.org/place/) are other great places to look for data.
# * This assignment requires you to convert and clean datafiles. Check out the discussion forums for tips on how to do this from various sources, and share your successes with your fellow students!
# 
# ## Example
# Looking for an example? Here's what our course assistant put together for the **Ann Arbor, MI, USA** area using **sports and athletics** as the topic. [Example Solution File](./readonly/Assignment4_example.pdf)

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd
import numpy as np

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')


# In[2]:

# file name
file_name = 'data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv'

# read data
df = pd.read_csv(file_name)

# convert date to datetime object
df['Date'] = pd.to_datetime(df['Date'])

# inspect data
df.head()

# filter data
df_TMAX = df[df['Element']=='TMAX'].reset_index(drop=True)
df_TMIN = df[df['Element']=='TMIN'].reset_index(drop=True)

# merge MAX and MIN dataframes
df = pd.merge(left=df_TMAX, right=df_TMIN, left_on=['ID','Date'], right_on=['ID','Date'],how='outer')
df = df.rename(columns={'Data_Value_x':'TMAX','Data_Value_y':'TMIN'})
df = df.drop(['Element_x','Element_y'],axis=1)

# extract day and month of dates
df['MonthDay'] = df['Date'].apply(lambda x: '{0:02}/{1:02}'.format(x.month,x.day))

# delete 02-29
df = df[df['MonthDay']!='02/29']

# convert data from tenth of degrees (C) to degrees (C)
df['TMAX'] = df['TMAX'].multiply(0.1)
df['TMIN'] = df['TMIN'].multiply(0.1)

# extract 2015
df_2015 = df[df['Date'].dt.year==2015]

# remove 2015
df = df[df['Date'].dt.year!=2015]

# group data
df_group = df.groupby('MonthDay')['TMAX','TMIN'].agg({'TMAX':np.max,'TMIN':np.min})
df_2015_group = df_2015.groupby('MonthDay')['TMAX','TMIN'].agg({'TMAX':np.max,'TMIN':np.min})

# sort by index
df_group = df_group.sort_index()
df_2015_group = df_2015_group.sort_index()
df_group = df_group.reset_index()
df_2015_group = df_2015_group.reset_index()

# 2015 data

# merge data of 2015 with grouped data
df_2015_group_merge = pd.merge(left=df_2015_group,right=df_group,
                               left_index=True,right_index=True,
                               suffixes=('_2015',''))


df_2015_group_merge['rec_max'] = (df_2015_group_merge['TMAX_2015']>df_2015_group_merge['TMAX']) * df_2015_group_merge['TMAX_2015']
df_2015_group_merge['rec_min'] = (df_2015_group_merge['TMIN_2015']<df_2015_group_merge['TMIN']) * df_2015_group_merge['TMIN_2015']

df_2015_max = df_2015_group_merge[df_2015_group_merge['rec_max']!=0]
df_2015_min = df_2015_group_merge[df_2015_group_merge['rec_min']!=0]


# In[4]:

# create figure
f, ax = plt.subplots(figsize=(16,10))

# plot each series (TMAX and TMIN)
df_group['TMAX'].plot(ax=ax,label='2005-2014 Highest',color='crimson',alpha=0.3)
df_group['TMIN'].plot(ax=ax,label='2005-2014 Lowest',color='dodgerblue',alpha=0.3)

# fill the area between the max data and min data
ax.fill_between(range(len(df_group['TMIN'])), 
                       df_group['TMIN'], df_group['TMAX'], 
                       facecolor='grey', 
                       alpha=0.10)

# plot the 2015 data
df_2015_max['rec_max'].plot(ax=ax,label='2015 new high',marker='o',linewidth=0,markersize=5,color='crimson')
df_2015_min['rec_min'].plot(ax=ax,label='2015 new low',marker='o',linewidth=0,markersize=5,color='dodgerblue')

# set x ticks
ax.set_xticks(range(0,365,31))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','July','Aug','Sept','Oct','Nov','Dec'],rotation=45)

# set texts
ax.set_ylabel('Daily temperature(°C)')
ax.set_xlabel('')
ax.set_title('Extreme Temperatures (2005-2014 and 2015) in Ann Harbor')
ax.legend(frameon=False)

f.savefig('TAD_plot');


# In[ ]:



