#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sqlite3 
import matplotlib.pyplot as plt
import matplotlib as mpl


# ## member.csv

# In[2]:


df_M= pd.read_csv("members_v3.csv")


# In[3]:


df_M.shape


# In[4]:


df_M.info()


# In[5]:


#check for duplicates
df_M.nunique() 


# In[6]:


#check for missing value
df_M.isnull().sum()


# In[7]:


#Replace empty gender value as No Data
df_M.gender=df_M.gender.replace(np.nan,'No Data')


# In[8]:


#Dropout the rest empty values
df_M=df_M.dropna()
df_M.isnull().sum()


# In[9]:


df_M.hist(color='deepskyblue',figsize=(10, 8))


# In[10]:


#check age values
df_M.bd.describe()


# In[11]:


#select reasonable range out of age
mask1=df_M.bd<=100                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
mask2=df_M.bd>0 
df_bd=df_M[mask1&mask2].bd


# In[12]:


df_M.bd.value_counts().head(30)
print('%% of Valid age values: %.5f%%'%(df_bd.count()/df_M.shape[0]))


# In[13]:


#Graphic:age in range 1-100
plt.figure(figsize=(7,5))
plt.rcParams['font.size']=15
plt.yticks(alpha=0)
plt.title('Distribution of Valid Age')
plt.xlabel('age')
plt.ylabel('users')
plt.bar(df_bd.value_counts().index.values,df_bd.value_counts(),color='deepskyblue')
df_M['bd']=df_bd


# In[14]:


#grouping by age base on the Distribution
#connect to the database
conn=sqlite3.connect('db')
cursor=conn.cursor()
print('opened database successfully')

# CREATE member_table 
sqlstr='''CREATE TABLE IF NOT EXISTS member(NSNO TEXT NOT NULL) '''
cursor.execute(sqlstr)
conn.commit()
df_M.to_sql('member', con=conn, if_exists='replace')
print('member Records created successfully')

groupage='''Select msno,BD,GENDER,CITY,registered_via,
case 
when BD<18 then '18-'
when BD between 18 and 20 then '18-20'
when BD between 21 and 25 then '21-25'
when BD between 26 and 30 then '26-30'
when BD between 31 and 35 then '31-35'
when BD between 36 and 40 then '36-40'
when BD between 41 and 50 then '41-50'   
when BD between 51 and 60 then '51-60'   
when BD between 61 and 70 then '61-70' 
when BD>71 then '71+' 
END AS Groupage
FROM member
GROUP BY MSNO'''
cursor.execute(groupage)
conn.commit()


# In[15]:


df_BD=pd.read_sql(groupage,conn)
BD=df_BD.Groupage.value_counts().sort_index(ascending=False)
BD=pd.DataFrame(BD)
BD['%']=np.array([(r)/BD.Groupage.sum() for r in BD.Groupage])


# In[16]:


#Graphic:bar of age group
color=['darkgrey','darkgrey','darkgrey','skyblue','skyblue','skyblue','deepskyblue','deepskyblue','skyblue','skyblue']
plt.rcParams['font.size']=15
plt.figure(figsize=(5,5))
plt.title('Distribution of Age group')
barh=plt.barh(BD.index,BD.Groupage,color=color)


# ## Transcation

# In[17]:


df_T=pd.read_csv('transactions.csv')
df_T.shape


# In[18]:


#check duplicates
df_T.nunique()


# In[19]:


df_T.isnull().sum()


# In[20]:


#select month valus form  membership_expire_date as DATE
df_T['DATE']=pd.DataFrame([d[0:6] for d in df_T['membership_expire_date'].astype(str)])


# In[21]:


#drop  duplicates user id  & keep the lastest transactions record
df_t=df_T.sort_values('transaction_date',ascending=(False)).drop_duplicates('msno')


# In[22]:


#calculate the total transaction records of each user (only counts once a month) 
df_T['t_times']=1
transactionstime=df_T.groupby(['msno','DATE'])['t_times'].count()
transactionstime=transactionstime.count('msno')
df_t=pd.merge(df_t,transactionstime,on='msno',how='left')
df_t.nunique()


# In[23]:


#sum the totalcharge price per user
totalcharges=df_T.groupby(['msno'])['plan_list_price'].sum()
totalcharges.name='totalcharges'
df_t=pd.merge(df_t,totalcharges,on='msno',how='left')


# In[24]:


plt.scatter(df_t.totalcharges,df_t.t_times)


# In[25]:


df_t.totalcharges.value_counts()


# In[26]:


plt.rcParams['font.size']=15
#df_t.hist(color='deepskyblue',figsize=(16, 10))


# In[27]:


#grouping the transaction time as user loyalty
tt=df_t.t_times.value_counts()
bins = [0,1,4,12,18,24,max(df_t.t_times)]
df_t['user_level'] = pd.cut(df_t.t_times, bins, labels= np.arange(1, len( bins) ) )
df_t['user_level'] = df_t['user_level'].astype('int')


# In[28]:


pybins = [0,34,35,36,37,38,39,40,41]
df_t['payment_method_id'] = pd.cut(df_t.payment_method_id,pybins, labels= np.arange(1, len(pybins) ) )
df_t['payment_method_id'] = df_t['payment_method_id'].astype('int')
payment_med=df_t['payment_method_id'].value_counts().sort_index(ascending=True)
df_t.payment_method_id.value_counts().sort_index()


# In[29]:


def DATE(df_t):
    if df_t['DATE']=='201703':
        df_t['contract']='0'
    else:
        df_t['contract']='1'
    return df_t
df_t=df_t.apply(DATE,axis=1)


# In[30]:


df_t.contract.value_counts()


# ## user log

# In[31]:


df_userlog=pd.read_csv('user_logs_v2.csv',header=0,encoding='utf-8-sig')
df_userlog.shape


# In[32]:


#create new table in sqlite
sqlstr='''CREATE TABLE IF NOT EXISTS userlog(NSNO TEXT NOT NULL) '''
conn=sqlite3.connect('db')
cursor=conn.cursor()
cursor.execute(sqlstr)
conn.commit()
df_userlog.to_sql('userlog', con=conn, if_exists='replace')
print('userlog Records created successfully')

#last 1 month user data
#find avg seconds played per user/ ratio songs played of song length
user_engagement='''Select msno,COUNT(msno) as log_counts, round(avg(num_unq),0) avg_unisongs_listened,
round(avg(total_secs),0) 'avg_totalseconds' ,
round((sum(num_25)*0.25+sum(num_50)*0.5+sum(num_75)*0.75+sum(num_985)*0.985+sum(num_100))/
sum(num_25+num_50+num_75+num_985+num_100),1)  listening_rate 
FROM userlog
GROUP BY MSNO'''
cursor.execute(user_engagement)
conn.commit()
df_usereng_1m=pd.read_sql(user_engagement,conn)
df_usereng_1m.columns
df_usereng_1m.describe().T


# In[33]:


#Total Second played
bins_song =[ -0.1,0,1800,3600,4400,7200,10800,max(df_usereng_1m['avg_totalseconds'])]
df_usereng_1m['secs_range'] = pd.cut(df_usereng_1m['avg_totalseconds'],bins_song, 
                                    labels= np.arange(len( bins_song)-1))
sec_total=df_usereng_1m['secs_range'].value_counts().sort_index(ascending=True)
sec_total.index=['0','0-1800s','1800-3600s','3600-4400s','4400-7200s','7200-10800s','10800s+']
print(sec_total)


# In[34]:


df_usereng_1m.secs_range.value_counts()


# In[35]:


#input kkbox train dataset
df_Train=pd.read_csv('train.csv',header=0,encoding='utf-8-sig')


# In[36]:


df_Train.shape


# ## Merge

# In[37]:


#%%merge ahttp://localhost:8888/notebooks/kkbox%20chrun%20prediction.ipynb#Mergell data by userID
df_All=pd.merge(df_Train,df_t,left_on='msno',right_on='msno')
df_All=pd.merge(df_All,df_usereng_1m,on='msno',how='left')
df_All=pd.merge(df_All,df_BD,on='msno',how='left')


# In[38]:


df_All.nunique()


# In[39]:


df_All.columns


# ### gender / age 

# In[40]:


df_All.gender=df_All.gender.replace(np.nan,'No Data')


# In[41]:


#Visualize the distribution of gender
plt.rcParams['font.family']='Microsoft YaHei'
plt.rcParams['font.size']=15
color=['darkgrey','deepskyblue','skyblue']
def datalabel(data):
   def inner_datalabel(pct):
       total=sum(data)
       val=int(round(pct*total/100.0))
       return '{p:.1f}% '.format(p=pct)
   return inner_datalabel
plt.subplot(121)
df_All.gender.value_counts().plot(kind='pie',title='User gender_All',fontsize=25,colors=color,figsize=(10, 10),autopct=datalabel(df_All.gender.value_counts()))
plt.subplot(122)
df_M.gender.value_counts().plot(kind='pie',title='User gender',fontsize=25,colors=color,figsize=(10, 10),autopct=datalabel(df_M.gender.value_counts()))


# In[42]:


#Visualize the distribution of gender
plt.rcParams['font.family']='Microsoft YaHei'
plt.rcParams['font.size']=15
color=['darkgrey','deepskyblue','skyblue']
df_BD.bd.notnull().value_counts().plot(kind='pie',fontsize=25,colors=color,figsize=(6, 6),autopct=datalabel(df_BD.bd.notnull().value_counts()))


# In[43]:


df_BD.bd.isnull().value_counts()


# In[44]:


df_BD['Groupage'].value_counts().sort_index()


# In[45]:


age_mapping = {
    '18-':1,'18-20':2,'21-25':3,'26-30':4,
    '31-35':5,'36-40':6,'41-50':7,'51-60':8,
    '61-70':9,'71+':10}
df_All['Groupage'] = df_All['Groupage'].map(age_mapping)


# In[46]:


#registered method
RM=df_M.registered_via.value_counts()/df_M.registered_via.count()
def createLabels(data):                  
    for item in data:
        height = item.get_height()
        plt.text(
            item.get_x()+item.get_width()/2., height*1.05, 
            '%.d%%' %int(height*100),ha = "center",va = "bottom",size=23)
plt.figure(figsize=(8,8))
plt.subplot(211)
plt.title('registered channels')
plt.xticks(RM.index,alpha=0)
plt.ylim(0,0.66)
plt.yticks(alpha=0)
plt.rcParams['font.size']=25
RMBAR=plt.bar(RM.index,RM,color='deepskyblue')
createLabels(RMBAR)
plt.subplot(212)
plt.ylim(0,0.9)
city=df_M.city.value_counts()/df_M.city.count()
plt.title('city')
plt.xticks(city.index,alpha=0)
plt.yticks(alpha=0)
citybar=plt.bar(city.index,city,color='deepskyblue')
createLabels(citybar)
plt.show()


# ### Transcation/loyalty/ payment

# In[47]:


#check Plan price & Plan days
df_All[['payment_plan_days','plan_list_price']].agg(['average','min', 'max']).T


# In[48]:


#colormap
def color(a):
    C=[]
    cmap = mpl.cm.get_cmap('cool')
    for c in np.arange(0.1,0.6,0.1):
        C.append(cmap(c))
    cmap1 = mpl.cm.get_cmap('Greys_r')
    for c1 in np.arange(0.5,a,0.1):
        C.append(cmap1(c1))
    return C 


# In[49]:


#Graphic: plan price & plan days top 10
def T(T,t):
    price10=T['plan_list_price'].value_counts()[0:100]
    labels=[f'NT${i}' for i in price10.index[0:100]]
    plt.figure(figsize=(12,8))
    plt.rcParams['font.size']=15
    plt.subplot(1,2,1)
    plt.title('PlanPrice'+str(t),fontsize=20)
    plt.pie(price10,labels=labels,autopct=datalabel(price10),colors=color(0.9))
    plan12=T['payment_plan_days'].value_counts()[0:100]
    label=[f'{a}days' for a in plan12.index[0:100]]
    plt.subplot(1,2,2)
    plt.title('PlanDay'+str(t),fontsize=20)
    plt.pie(plan12,labels=label,autopct=datalabel(plan12),colors=color(1.2))
    plt.show()   
T(df_t,'_pre user')
T(df_All,'_exclude free trail')


# In[50]:


bin_day =[-0.1,7,30,max(df_All['payment_plan_days'])]
df_All['planday_group'] = pd.cut(df_All['payment_plan_days'],bin_day, 
                                     labels= np.arange(1,len( bin_day)) )
group_day=df_All['planday_group'].value_counts().sort_index(ascending=True)
print(group_day/sum(group_day))


# In[51]:


#plan day vs plan price
plt.scatter(df_All.plan_list_price,df_All.planday_group)


# In[52]:


df_tt=df_t[df_t.payment_plan_days!=7]
df_tt=df_tt[df_tt.plan_list_price!=0] 


# In[53]:


def Bar(X):
    total=0
    for i in  X:
        total+=i
    for c,y in zip(X.index,X):
      plt.text(c,y+max(X)*0.05,str('{:,d}\n({:.2f}%)'.format(y,(y/total*100))),ha='center')


# In[54]:


#Graphic:loyalty
def show(x,word):
    user_level=x['user_level'].value_counts().sort_index(ascending=True)
    df_All.user_level.value_counts().sort_index()
    plt.figure(figsize=(10,5))
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['font.size']=20
    plt.title('User Loyalty '+str(word),size=28)
    times=['1(New)','2-4','5-12','11-18','18-24','24+']
    plt.yticks(alpha=0)
    plt.ylim(0,max(user_level)*1.25)
    plt.xlabel('transaction times')
    ttbar=plt.bar(user_level.index,user_level,color='skyblue',tick_label=times)
    Bar(user_level)
show(df_tt,'\n(Exclude free trail)')
show(df_t,'')


# In[55]:


#auto_Renew
df_All['is_auto_renew'].value_counts()/df_All['is_auto_renew'].count()


# In[56]:


#Graphic:payment method 
pm=df_t['payment_method_id'].value_counts().sort_index(ascending=False)
plt.rcParams['font.size']=18
plt.figure(figsize=(20,5))
plt.title('payment_method',size=35)
plt.xlabel('id')
plt.ylabel('user_counts')
plt.xticks(pm.index)
plt.yticks(alpha=0)
plt.ylim(0,max(pm)*1.4)
pmbar=plt.bar(pm.index,pm,color='deepskyblue')
Bar(pm)


# ### ratio/total secs

# In[57]:


df_userlog.isnull().sum()


# In[58]:


df_userlog.nunique()


# In[59]:


df_All.listening_rate.value_counts().sort_index()


# In[60]:


#ratio of songs played
df_All.listening_rate=df_All.listening_rate.astype(str)
Usere=df_All.listening_rate.value_counts().sort_index()
Usere.index=['0','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1']
Usere/df_All.listening_rate.count()
Usere


# In[61]:


#Graphic:ratio of songs played
plt.figure(figsize=(8,6))
plt.title('Ratio of Songs played',size=20)
rates=['0','0.3',' 0.4',' 0.5',' 0.6',' 0.7',' 0.8','0.9',' 1']
plt.yticks(alpha=0)
#plt.xticks(rates)
plt.xlabel('songs played %')
plt.ylim(0,440000)
uebar=plt.bar(Usere.index,Usere,color='deepskyblue',tick_label=rates,alpha=0.6)
total=0
for i in  Usere:
    total+=i
for c,y in zip(Usere.index,Usere):
  plt.text(c,y+max(Usere)*0.05,str('{:.2f}%'.format((y/total*100))),ha='center')


# In[62]:


#find avg seconds played per user
df_userlog.groupby('msno')['total_secs'].mean()
unisongs=df_usereng_1m.avg_unisongs_listened.astype(int)
unisongs.value_counts()[0:20]
unisongs.count()


# In[63]:


#average song length played per users
tlist=['num_25', 'num_50', 'num_75', 'num_985', 'num_100','num_unq', 'total_secs']
Userlog_avg=df_userlog.groupby('msno')[tlist].mean().astype(int).reset_index()
Userlog_avg.total_secs.describe()
print(Userlog_avg.num_unq.describe())
print(Userlog_avg['num_unq'].mean())
Userlog_avg['total_secs'].mean() 


# In[64]:


#Graphic: check unique distribution
def Plot(nn,n):
    plt.figure(figsize=(6,6))
    plt.title('Length of Songs played')
    plt.xlabel('average_times')
    plt.ylabel('users')
    N=0  
    for Col in Userlog_avg.columns[nn:n+1]:  
      num=Userlog_avg[Col].value_counts().sort_index()[0:15]
      plt.plot(num,'--o',label=Col,color=colors[N],linewidth=2)
      N+=1
      plt.legend()
colors=color(0.7)
Plot(1,5)


# In[65]:


def unisong(n,r,z):
    plt.figure(figsize=(16,6))
    
    plt.xlabel('songs')
    plt.ylabel('user counts')
    plt.subplot(1,3,1)
    plt.title('Unique songs played_'+str(n))
    plt.bar(Userlog_avg['num_unq'].value_counts().sort_index()[0:n].index,
            Userlog_avg['num_unq'].value_counts().sort_index()[0:n].values,
             label='num_unq',color='deepskyblue')
    plt.axvline(x=23.5,c='k',label='average songs')
    plt.subplot(1,3,2)
    plt.title('Unique songs played_'+str(r))
    plt.bar(Userlog_avg['num_unq'].value_counts().sort_index()[0:r].index,
            Userlog_avg['num_unq'].value_counts().sort_index()[0:r].values,
             label='num_unq',color='deepskyblue')
    plt.axvline(x=23.5,c='k',label='average songs')
    plt.subplot(1,3,3)
    plt.title('Unique songs played_'+str(z))
    plt.bar(Userlog_avg['num_unq'].value_counts().sort_index()[0:z].index,
            Userlog_avg['num_unq'].value_counts().sort_index()[0:z].values,
             label='num_unq',color='deepskyblue')
    plt.axvline(x=23.5,c='k',label='average songs')
    plt.legend()

unisong(50,100,150)


# In[66]:


df_usereng_1m.columns


# In[67]:


df_usereng_1m.isnull().sum()


# In[68]:


#Graphic:total sec
plt.figure(figsize=(8,6))
plt.title('Total seconds played',size=20)

plt.yticks(alpha=0)
plt.ylim(0,350000)
plt.xlabel('sec')
plt.ylabel('user_counts')
secbar=plt.bar(sec_total.index,sec_total,color='deepskyblue',alpha=0.5)
plt.axvline(x=3.23,c='k',label='average songs')
total=0
for i in  sec_total:
    total+=i
for c,y in zip(sec_total.index,sec_total):
  plt.text(c,y+max(sec_total)*0.05,str('{:,d}\n({:.2f}%)'.format(y,(y/total*100))),ha='center')


# ### Churn -Data Visualization 

# In[69]:


df_All.isnull().sum()


# ##Churn

# In[70]:


#check the churn rate
plt.figure(figsize=(8,8))
plt.rcParams['font.size']=18
data=df_All['is_churn'].value_counts()
labels=['not-churn','churn']  
colors=['darkgrey','deepskyblue']        
explode= [0,0.2]
plt.pie(data, labels=labels, autopct=datalabel(data),
        colors=colors,startangle=30,explode=explode)                                                                                          
plt.show()


# In[71]:


df_All['totalcharges'].agg(['average','max','min','count'])


# In[72]:


bin_day =[-0.1,7,30,max(df_All['payment_plan_days'])]
df_All['planday_group'] = pd.cut(df_All['payment_plan_days'],bin_day, 
                                     labels= np.arange(1,len( bin_day)) )
group_day=df_All['planday_group'].value_counts().sort_index(ascending=True)
print(group_day/sum(group_day))


# In[73]:


df_All[['bd','avg_unisongs_listened']].plot.scatter(x ='bd',y='avg_unisongs_listened')


# In[74]:


df_All[['bd','t_times']].plot.scatter(x ='bd',y='t_times')


# In[75]:


df_All[['plan_list_price','totalcharges']].plot.scatter(x ='plan_list_price',y='totalcharges')


# In[76]:


import seaborn as sns
def DTB(col,title):
    plt.figure(figsize=(6,6))
    plt.rcParams['font.size']=10
    ax = sns.kdeplot(df_All[col][(df_All["is_churn"] == 1)],
                    color="deepskyblue", shade= True)
    ax = sns.kdeplot(df_All[col][(df_All["is_churn"] == 0)],
                     ax =ax,color="Gray", shade = True)   
    ax.legend(["Churn","Not Churn"],loc='upper right')
    ax.set_ylabel('Density')
    ax.set_xlabel(title,size=16)
    ax.set_title('Distribution of '+str(title)+' by churn',size=16)
DTB('totalcharges','Total Charges')


# In[77]:


DTB('plan_list_price','Monthly Price')


# In[78]:


DTB('avg_unisongs_listened','unique songs played')


# In[79]:


DTB('payment_method_id','membership_expire_date')


# In[80]:


plan='''Select plan_list_price,
case 
    when payment_plan_days=30 then '30'
    when payment_plan_days=7 then '7'
    else 'oth'
END AS plan_days,
count(msno) num ,sum(is_churn) ,cast(sum(is_churn) as float)/cast(count(msno) as float) as churn
FROM user
GROUP BY plan_list_price,plan_days
order by sum(is_churn) desc,churn desc '''
cursor.execute(plan)
conn.commit()
df_plan=pd.read_sql(plan,conn)


# In[81]:


#Plan price
#select top 16 where total churn >100
Plan=df_plan.sort_values('sum(is_churn)',ascending=False)[0:20]
Plan=Plan.groupby('plan_list_price').mean()
Plan=Plan.sort_index()
Plan.num=Plan.num.astype(int)
Plan.churn=Plan.churn.astype(float)
Plan.index=Plan.index.astype(str)
Plan['Notchurn']=Plan.num-Plan['sum(is_churn)'].astype(int)
Plan['sum(is_churn)'].astype(int)


# In[82]:


fig = plt.figure(figsize=(40,20))
ay1 = fig.add_subplot(111)
plt.rcParams['font.size']=40
plt.title('Churn by Plan price',size=50) 
plt.yticks(alpha=0)
plt.xticks(rotation=90,size=38)
plt.ylim(0,max(Plan.num)*1.8)
plt.bar(Plan.num.index,Plan.Notchurn,color='darkgrey',alpha=0.6)
plt.bar(Plan.num.index,Plan['sum(is_churn)'],bottom=Plan.Notchurn,color='skyblue',label='Churn        ')
plt.legend(bbox_to_anchor=(1, 0.935)) 
ay2 = ay1.twinx()
plt.ylim(-min(Plan.churn)*1.7,max(Plan.churn)*1.3)
plt.yticks(alpha=0)
plt.plot(Plan.index,Plan.churn,'--o',color='deepskyblue',linewidth='5',markersize="30",label='Churn_rate')
plt.legend()
for c,y in zip(Plan.churn.index,Plan.churn):
  plt.text(c,y+max(Plan.churn)*0.05,str('{:.1f}%'.format(y*100)),ha='center')


# In[83]:


df=df_All[['totalcharges','is_churn']]
totalcharges=df.totalcharges.value_counts().sort_index()


# In[84]:


#churn variables
df_all_na=df_All.replace(np.nan,'nodata')
df_All.isnull().sum()/df_All.shape[0]

for col in df_All.columns[7:]:
 print(pd.crosstab(df_All['is_churn'],df_All[col]))


# In[85]:


df_All.columns


# In[86]:


#Graphic
def Column(col,TITLE,loc,l,*a):
    xticks=[]
    for i in a:
     xticks.append(str(i))
    C=pd.crosstab(df_All['is_churn'],df_All[col]).columns.astype(str)
    V=pd.crosstab(df_All['is_churn'],df_All[col]).values
    np.sum(V,axis=0)
    Y=np.divide(V[1],np.sum(V,axis=0))
    actualNum=df_All[col].sort_index().value_counts()
    fig = plt.figure(figsize=(8,6))
    ay1 = fig.add_subplot(111)
    plt.rcParams['font.size']=15
    plt.title(TITLE,size=25)
    plt.yticks(alpha=0)
    plt.xticks(size=18)
    plt.ylim(0,max(actualNum)*2.1)
    plt.bar(C,V[0],color='darkgrey',tick_label=xticks,alpha=0.6)
    plt.bar(C,V[1],color='skyblue',bottom=V[0],tick_label=xticks,alpha=0.6,label='Churn      ')
    plt.legend(bbox_to_anchor=(l, 0.93))
    ay2 = ay1.twinx()
    plt.ylim(-min(Y)*0.7,max(Y)*1.2)
    plt.yticks(alpha=0)
    plt.plot(C,Y,'--o',color='deepskyblue',linewidth='3',markersize="20",label='Churn rate')
    plt.legend(loc=loc)
    for i in range(len(Y)):
     print('%s: %.2f %%'%(C[i],Y[i]))
     for c,y in zip(C,Y):
      plt.text(str(c),y+max(Y)*0.05,str('{:.1f}%'.format(y*100)),ha='center')


# In[87]:


df_All.listening_rate=df_All.listening_rate.replace(np.nan,0)


# In[88]:


df_All.listening_rate.value_counts()


# In[89]:


Column('listening_rate','Churn by ratio of songs played','best',1,'0','0.3',' 0.4',' 0.5',' 0.6',' 0.7',' 0.8','0.9',' 1')     


# In[90]:


Column('planday_group','Churn by Plan day','best',1,'7days','8-30days','30days+')


# In[91]:


Column('is_cancel','Churn by cancel records','upper left',0.31,'NO','YES')


# In[92]:


df_All.DATE.value_counts()


# In[93]:


Column('contract','Churn by contract duration','upper left',0.31,'Subscribe','Expire')


# In[94]:


df_All.secs_range.value_counts().sort_index()


# In[95]:


pd.crosstab(df_All['is_churn'],df_All['secs_range'])


# In[96]:


df_All.secs_range=df_All.secs_range.replace(np.nan,0)


# In[97]:


Column('secs_range','Chunr by total seconds','best',1,'0','0-1800s','1800-3600s','3600-4400s','4400-7200s','7200-10800s','10600s+')


# In[98]:


Column('registered_via','Churn by registered channels','best',1,'3','4','7','8','13')


# In[99]:


Column('payment_method_id','Churn by payment method','best',1,1,2,3,4,5,6,7,8)


# In[100]:


Column('Groupage','Churn by age groups','best',1,'18-', '18-20', '21-25', '26-30', '31-35', '36-40', '41-50', '51-60',
       '61-70', '71+')


# In[101]:


Column('user_level','Churn by user loyalty','best',0.99,'1(New)','2-4','5-12','11-18','18-24','24+')


# In[102]:


df_All.user_level.value_counts()


# In[103]:


df_All.columns


# In[104]:


df_All.isnull().sum()


# In[105]:


print('Without listening records%%\n%.2f%%'%(df_All.secs_range.isnull().sum()/df_All.is_churn.count()))


# In[106]:


print('Churn rate without listening records in month\n%.4f%%'%(df_All[df_All.secs_range.isnull()].is_churn.sum()/df_All[df_All.secs_range.isnull()].is_churn.count()))


# In[107]:


df_All.head(20)


# In[108]:


df_All=df_All.replace(np.nan,0)


# In[109]:


for col in df_All.columns[-11:-1]:
 print(col)
 print(df_All[col].unique())


# In[110]:


df_All.isnull().sum()


# In[111]:


df_All.listening_rate=df_All.listening_rate.replace('nan',0)


# In[112]:


df_All.secs_range=df_All.secs_range.fillna(0)


# In[113]:


df_Allml=df_All.drop(['bd','transaction_date','membership_expire_date','city','gender','log_counts',
                    't_times','avg_totalseconds','payment_plan_days','msno','avg_unisongs_listened','actual_amount_paid'],axis=1)


# In[114]:


df_Allml.head(10)


# ### Machine Learning

# In[116]:


from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
df_All.columns
df_All=df_All.replace(np.nan,-1) 
X=df_Allml.drop(columns = ['is_churn','is_cancel','is_auto_renew'],axis=1)
y=df_Allml.is_churn


# In[117]:


# Scaling all the variables to a range of 0 to 1
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features


# In[118]:


X.head(15)


# In[119]:


logistic=linear_model.LogisticRegression()
Xtrain,Xtest,ytrain,ytest=tts(X,y,test_size=0.4,random_state=100)


# In[120]:


Xtrain.hist(color='deepskyblue',figsize=(10, 8))


# In[121]:


Xtest.hist(color='deepskyblue',figsize=(10, 8))


# In[122]:


logistic.fit(Xtrain,ytrain)


# In[123]:


for i in range(len(Xtrain.columns)):
 print('coef of %s : %.8f'%(Xtrain.columns[i],logistic.coef_[0][i]))


# In[124]:


Xtrain.columns


# In[125]:


pred=logistic.predict(Xtest)
print('logistic Accuracy: ',logistic.score(Xtest,ytest))


# In[126]:


from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
report = classification_report(ytest,pred)
print(report)


# In[127]:


from sklearn.metrics import roc_curve
prob = logistic.predict_proba(Xtest)[:,1]
fpr, tpr, thresholds = roc_curve(ytest,prob)
plt.figure(figsize=(8,6))
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr, tpr, label='Logistic Regression',color = "skyblue")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve',fontsize=16)
plt.show()


# In[128]:


#knn


# In[129]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 20) 
knn.fit(Xtrain,ytrain)
predicted_y = knn.predict(Xtest)
accuracy_knn = knn.score(Xtest,ytest)
print("KNN accuracy:",accuracy_knn)


# In[ ]:


print(classification_report(ytest, predicted_y))


# In[ ]:


df_T.transaction_date.describe()


# In[ ]:





# In[ ]:




