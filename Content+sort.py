
# coding: utf-8

# ## Get hotel parameters 

# In[142]:

import urllib.request, json
import requests
import vertica_python as vp
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot
from matplotlib import pylab as plt
from __future__ import generators


# In[4]:

def ResultIter(cursor, arraysize=1000):
    'An iterator that uses fetchmany to keep memory usage down'
    while True:
        results = cursor.fetchmany(arraysize)
        if not results:
            break
        for result in results:
            yield result


# In[5]:

basic_conn_info = {'host': 'stat-vertica-db.ostrovok.ru',
            'port': 5433,
            'user': 'readonly_user',
            'password': 'readonly_user_vertica_prod',
            'database': 'verticaprod',
            # 10 minutes timeout on queries
            'read_timeout': 600,
            # default throw error on invalid UTF-8 results
            'unicode_error': 'strict',
            # SSL is disabled by default
            'ssl': False}


# In[10]:

storage=[]
with vp.connect(**basic_conn_info) as connection:
        cur = connection.cursor()
        cur.execute("with c as ( select currency_to, actual_date, avg_rate as rate FROM analytics.currency_history ) ,t as ( select otahotel_id, round(coalesce((price) / c.rate::float * cc.rate::float/length_of_stay, 1),3) as price_converted from weblogs2.hc_displays hc left join c on c.actual_date = hc.datetime::date and upper(hc.currency) = c.currency_to left join c cc on cc.actual_date = hc.datetime::date and cc.currency_to = 'RUB' where datetime>=now()::date - '6 month'::interval and length_of_stay>0 ), hotel_prices as ( select otahotel_id, avg(price_converted) as price from t group by 1 ), content_scores as ( SELECT distinct otahotel_id, first_value(hotel_gallery_score) OVER(PARTITION BY otahotel_id ORDER BY created_at desc) AS last_gallery_score, first_value(hotel_gallery_filled) OVER(PARTITION BY otahotel_id ORDER BY created_at desc) AS last_gallery_filled, first_value(amenities_score) OVER(PARTITION BY otahotel_id ORDER BY created_at desc) AS last_amenities_score, first_value(hotel_gallery_photos_good_size) OVER(PARTITION BY otahotel_id ORDER BY created_at desc) AS last_gallery_photos_good_size FROM content.content_score ) select distinct sn.otahotel_id, country_en, city_en, last_value (sn.star_rating) over (PARTITION BY sn.otahotel_id ORDER BY sn.updated_at asc) as star_rating, last_value (sn.kind) over (PARTITION BY sn.otahotel_id ORDER BY sn.updated_at asc) as kind, last_gallery_score, last_gallery_filled, last_amenities_score, last_gallery_photos_good_size, price from content.content_snapshot sn left join content_scores sc on sn.otahotel_id = sc.otahotel_id left join public.otahotel o on o.id = sn.otahotel_id left join hotel_prices p on p.otahotel_id = sn.otahotel_id where sn.kind is not null and sn.star_rating is not null and price is not null ")
        for row in ResultIter(cur,arraysize=10000):
            storage.append(row)


# In[32]:

df = pd.DataFrame(storage, columns=['otahotel_id', 'country', 'city', 'star_rating', 'kind', 'gallery_score', 'gallery_filled', 'amenities_score', 'gallery_photos_good_size', 'price'] )
df


# In[196]:

df.to_csv('content_data', index = False)


# In[203]:

storage=[]
with vp.connect(**basic_conn_info) as connection:
        cur = connection.cursor()
        cur.execute("select otahotel_id, count(id) from weblogs2.hc_displays where datetime>=now()::date - '6 month'::interval and length_of_stay>0 and type = 'serp' and region_id in (with t as ( select region_id, count(id) as cnt from weblogs2.hc_displays group by 1 ) select region_id from t order by cnt desc limit 100) group by 1")
        for row in ResultIter(cur,arraysize=10000):
            storage.append(row)


# In[204]:

shows = pd.DataFrame(storage, columns=['otahotel_id', 'shows'] )
shows


# In[205]:

shows.to_csv('content_shows', index = False)


# In[206]:

shows = pd.read_csv('content_shows', encoding = 'koi8_r', index_col= False)
shows


# In[197]:

df = pd.read_csv('content_data', encoding = 'koi8_r', index_col= False)


# In[198]:

df


# In[92]:

df2 = pd.DataFrame(df.groupby(['country', 'city'])['price'].mean()).rename(columns = {'price':'region_price'}).reset_index()


# In[93]:

df2.head()


# In[209]:

df_main = pd.merge(df, df2, how='left', on=['country','city'])


# In[210]:

df3 = pd.DataFrame(df.groupby(['country', 'city', 'star_rating', 'kind'])['price'].mean()).rename(columns = {'price':'group_price'}).reset_index()


# In[211]:

df_main = pd.merge(df_main, df3, how='left', on=['country','city', 'star_rating', 'kind'])


# In[212]:

df_main = pd.merge(df_main, shows, how = 'inner', on=['otahotel_id'])


# In[213]:

df_main


# In[215]:

df_main['price_koeff'] = df_main['group_price']/df_main['region_price']


# In[252]:

g = df_main.groupby(['country', 'city'])


# In[270]:

l = []
for key, region_df in g:
    n_samples = 1500
    region_df = region_df.dropna()
    X = region_df[["price_koeff"]]
    if len(X)<6:
        continue
    y_pred = KMeans(n_clusters= 5, random_state=100).fit_predict(np.reshape(X, len(X), 1))
    region_df['cluster'] = y_pred
    l.append(region_df)
final_df = pd.concat(l)


# In[303]:

final_df = final_df.groupby(['country', 'city']).apply(lambda x: x.sort_values(['shows'], ascending = False).reset_index(drop=True))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[167]:

X = df_main[df_main.city =='Moscow'][["price_koeff"]]


# In[187]:

n_samples = 1500

y_pred = KMeans(n_clusters=6, random_state=100).fit_predict(X)


# In[188]:

plt.figure(figsize=[15, 12])
matplotlib.pyplot.scatter(X.index, X,c=y_pred)


# In[189]:

plt.show()

