
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


# In[155]:


#导入数据
IF = pd.read_excel('data_IH.xlsx',sheet_name = 'ih')


# In[196]:


# IF_ = IF['收盘价(元)'].values
IF_ = pd.read_csv('wden_ih.csv',header=None)


# In[197]:


IF_close = (np.array(IF_[1:])-np.array(IF_[0:-1]))
IF_close = IF_close/np.linalg.norm(IF_close)


# In[198]:


plt.plot(IF_close)


# In[140]:


def FPE(train, a):
    k = len(a)
    n= len(train)
    sigma = 0.0
    S = 0.0
    #calculate sum of residual square
    for  t in range(k,n):
        model = np.sum([a[i]*train[t-i-1] for i in range(0,k)])
        resid = train[t]-model
#         print(model,train[t],resid)
        S = S + resid**2
    fpe = (n+k+1)*S/(n-k-1)
    return(fpe)


# In[9]:


def find_k(train):
    mult_a = []
    mult_fpe = np.empty(8)
    for i in range(2,10):
        a = burgs(train,i)
        mult_a.append(a)
        mult_fpe[i-2] = FPE(train,a)
    
    argmin = np.where(mult_fpe==min(mult_fpe))[0][0]
#     return(argmin+2,mult_a[argmin],mult_fpe[argmin])
    return(argmin+2,mult_a[argmin])
#     return(argmin+2)


# In[51]:


find_k(IF_close[12380:12420])


# In[211]:


#已弃用
def yule_walker(x,k):
    N = len(x)
    m = np.mean(x)
    p_hat = np.empty(k+1)
    for i in range(0,k+1):
        prod = np.multiply(x[i+1:N]-m,x[1:N-i]-m)
        p_hat[i] = np.sum(prod)/N
    
    A = np.fromfunction(lambda i,j: 
                        np.where(i<j,p_hat[j-i],np.where(i>j,p_hat[i-j],p_hat[0])),
                        (k,k),dtype=int)
    b = p_hat[1:k+1]
    a = np.linalg.solve(A,b)
    return(a)#return a1,a2,...,ak


# In[141]:


def sigma_2(x,p):
    t=len(x)
    data=x.T
    data1=np.mat(data)
    g=p+1
    e=np.mat(np.zeros((p+1,t)))
    b=np.mat(np.zeros((p+1,t)))
    ss=np.mat(np.zeros((1,p+1)))
    aa=np.mat(np.zeros((p,p)))
    k1=np.zeros(p)
    e[0,:]=data
    b[0,:]=data
    ss[0,0]=np.mean(np.multiply(data1,data1),axis=1)
    for k in range(0,p):
        if k==0:
            h1=e[k,k+1:t]
            h2=b[k,k:t-1]
            hh=np.sum(np.multiply(h1,h2))
            hhh=np.sum(np.multiply(h1,h1)+np.multiply(h2,h2))
            k1[k]=-2*hh/hhh
        else:
            h11=e[k,k:t]
            h22=b[k,k-1:t-1]
            hh1=np.sum(np.multiply(h11,h22))
            hhh1=np.sum(np.multiply(h11,h11)+np.multiply(h22,h22))
            k1[k]=-2*hh1/hhh1

        if k>0:
            for i in range(0,k):
                aa[k,i]=aa[k-1,i]+k1[k]*aa[k-1,k-i-1]
        aa[k,k]=k1[k]
        
        for r in range(0,t):
            if r==0:
                e[k+1,r]=e[k,r]
                b[k+1,r]=k1[k]*e[k,r]
            else:
                e[k+1,r]=e[k,r]+k1[k]*b[k,r-1]
                b[k+1,r]=b[k,r-1]+k1[k]*e[k,r]
        ss[0,k+1]=(1-k1[k]**2)*ss[0,k]
    return(ss[0][0,-1],-np.array(aa[p-1,:])[0])
#     return(ss[0][0,-1])


# In[142]:


# a = sigma_2(IF_close[0:40],2)[1]
# FPE(IF_close[4760:4800],a),a
sigma_2(IF_close[100:140],2)


# In[204]:


def burgs(x,k):
    N = len(x)
    f = x[1:N]
    b = x[0:N-1]
    mu = -(2*b.T@f)/(f.T@f+b.T@b)
    a = np.array([1,0])
    ss = np.mean(np.multiply(x,x))
    for i in range(1,k+1):
        J = np.fromfunction(lambda m,n: np.where(m+n==i,1,0), (i+1,i+1))
        f = np.array([a.T@J@x[h-i:h+1] for h in range(i,N)])
        b = np.array([a.T@x[h-i:h+1] for h in range(i,N)])
        mu = -(2*b.T@f)/(f.T@f+b.T@b)
        a = a + mu*J@a
        a = np.concatenate([a,[0]])
        ss = (1-mu**2)*ss
#     return(-a[1:-1],ss[0][0])
    return(-a[1:-1])


# In[145]:


def spectral_density(sigma_a, a, f):
    k = len(a)
    S = 2*sigma_a/np.power(abs(1-np.sum([a[h-1]*np.exp(-2*1j*np.pi*f*h) for h in range(1,k+1)])),2)
    return(S)


# In[199]:


#收益率计算
#2016年IF_close
#IH
index =(IF['日期'].dt.year == 2016)
data = IF_close[index[:-1]]
#IC
# data = IF_close[0:8764]
#IF
# data = IF_close[0:9864]
# profit = pd.DataFrame({'profit_rate':[],'sell':[],'sell_time':[],'buy':[],'buy_time':[]})
profit_rate = []
sell = []
sell_time = []
buy = []
buy_time = []
diff = []
acc_profit2016 = 0.0

MA10 = np.zeros(len(data))
freq_list = np.empty(len(data))
cyc_list = np.empty(len(data))
n = 40
k = 2
f = np.linspace(0.01,0.99,200)

for i in range(n,len(data)):
#     k,a,fpe = find_k(data[i-n+1:i+1])
    a,sigma = burgs(data[i-n:i],k)
    print(sigma,a)
#     sigma = sigma_2(data[i-n:i],k)[0]
    y = map(lambda x: spectral_density(sigma,a,x),f)
    y = list(y)
    freq_list[i] = f[np.where(y==max(y))][0]
    cyc_list[i] = 1/freq_list[i]
    if(i >= n+10):
        MA10[i] = min(max(np.mean(cyc_list[i-10:i]),5),30)
        if((MA10[i]>=10) & (MA10[i-1]<10)):#上穿买入
#         if((cyc_list[i]>=10) & (cyc_list[i-1]<10)):
#             print('buy')
            buy_ = IF['收盘价(元)'].values[i-1]
            buy_time_ = i
            if(len(sell)>0):
                buy.append(-sell_)
                buy_time.append(sell_time_)
                sell.append(-buy_)
                sell_time.append(buy_time_)
                profit_rate.append(sell_/buy_-1)
                acc_profit2016 = acc_profit2016 + sell_- buy_
        if((MA10[i]<=10)& (MA10[i-1]>10)):#下穿卖出
#         if((cyc_list[i]<=10) & (cyc_list[i-1]>10)):
#             print('sell')
            sell_ = IF['收盘价(元)'].values[i-1]
            sell_time_ = i
            buy.append(buy_)
            buy_time.append(buy_time_)            
            sell.append(sell_)
            sell_time.append(sell_time_)
            profit_rate.append(sell_/buy_-1)
            acc_profit2016 = acc_profit2016 + sell_- buy_


# In[152]:


profit2016 = pd.DataFrame({'profit_rate':profit_rate,'sell':sell,'sell_time':sell_time,
                           'buy':buy,'buy_time':buy_time,
                           'cum_profit_rate':np.cumsum(profit_rate)})
acc_profit2016,np.sum(profit2016['profit_rate']),len(profit2016)
#2016累计收益点数,累计收益率,交易次数


# In[66]:


np.sum(profit2016['profit_rate']<0)


# In[68]:


hh = profit2016.iloc[np.where(profit2016['profit_rate']<0)[0]]
np.mean(hh['profit_rate'])
# np.mean(profit2016['profit_rate'])


# In[73]:


fig = plt.figure()
data = IF['收盘价(元)'].values[5000:6000]
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                   xticklabels=[], ylim=(min(data)-5, max(data)+5))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                   ylim=(5, 40))
ax1.plot(data)
ax2.plot(MA10[5000:6000],'r')


# In[153]:


# nn = np.concatenate([[0],neg_profit_rate])
plt.plot(profit2016['sell_time'],(profit2016['profit_rate']+1).cumprod()*IF['收盘价(元)'].values[0]
         ,'b',IF['收盘价(元)'].values[0:7964],'r')
plt.rcParams['figure.figsize'] = (8, 4) 
plt.legend(['净值增长','标的价格'])
plt.title('2016IC: 净值增长 vs 标的价格')


# In[175]:


#2016净值曲线
data = IF['收盘价(元)'].values[0:9864]
aa= np.ones(len(data))
net_value = np.empty(len(data))
for i in range(len(profit2016)):
    b_time = profit2016['buy_time'].iloc[i]
    s_time = profit2016['sell_time'].iloc[i]
    price = data[b_time:s_time]
    growth_rate = (price[1:]-price[0:-1])/price[0:-1]
    aa[b_time+1:s_time] = 1.0+growth_rate

for j in range(len(data)):
    net_value[j] = aa[0:j].prod()
    
# plt.plot(net_value)
plt.plot(net_value)


# In[103]:


#收益率计算
#2017年IF_close[0：10854]
#IC
index =(IF['日期'].dt.year == 2017)
data = IF_close[index[0:-1]]
#IH
# data = IF_close[7964:20064]
#IF
# data = IF_close[9864:21964]
# profit = pd.DataFrame({'profit_rate':[],'sell':[],'sell_time':[],'buy':[],'buy_time':[]})
profit_rate = []
sell = []
sell_time = []
buy = []
buy_time = []
diff = []
acc_profit2017 = 0.0

MA10 = np.zeros(len(data))
freq_list = np.empty(len(data))
cyc_list = np.empty(len(data))
n = 40
k = 2
f = np.linspace(0.01,0.99,200)

for i in range(n,len(data)):
#     k,a,fpe = find_k(data[i-n+1:i+1])
    a = burgs(data[i-n:i],k)
#     fpe = FPE(data[i-n+1:i+1],a)
    sigma = sigma_2(data[i-n:i],k)[0]
    y = map(lambda x: spectral_density(sigma,a,x),f)
    y = list(y)
    freq_list[i] = f[np.where(y==max(y))][0]
    cyc_list[i] = 1/freq_list[i]
    if(i >= n+10):
        MA10[i] = min(max(np.mean(cyc_list[i-10:i]),5),30)
        if((MA10[i]>=10) & (MA10[i-1]<10)):#上穿买入
#         if((cyc_list[i]>=10) & (cyc_list[i-1]<10)):
#             print('buy')
            buy_ = IF['收盘价(元)'].values[9864+i-1]
            buy_time_ = i
            if(len(sell)>0):
                buy.append(-sell_)
                buy_time.append(sell_time_)
                sell.append(-buy_)
                sell_time.append(buy_time_)
                profit_rate.append(sell_/buy_-1)
                acc_profit2017 = acc_profit2017 + sell_- buy_
        if((MA10[i]<=10)& (MA10[i-1]>10)):#下穿卖出
#         if((cyc_list[i]<=10) & (cyc_list[i-1]>10)):
#             print('sell')
            sell_ = IF['收盘价(元)'].values[9864+i-1]
            sell_time_ = i
            buy.append(buy_)
            buy_time.append(buy_time_)            
            sell.append(sell_)
            sell_time.append(sell_time_)
            profit_rate.append(sell_/buy_-1)
            acc_profit2017 = acc_profit2017 + sell_- buy_


# In[104]:


profit2017 = pd.DataFrame({'profit_rate':profit_rate,'sell':sell,'sell_time':sell_time,
                           'buy':buy,'buy_time':buy_time,
                           'accum_profit_rate':np.cumsum(profit_rate)})
acc_profit2017,np.sum(profit2017['profit_rate']),len(profit2017) 
#2017累计收益点数,累计收益率,交易次数


# In[105]:


np.sum(profit2017['profit_rate']>0)


# In[101]:


hh = profit2017.iloc[np.where(profit2017['profit_rate']>0)[0]]
np.mean(hh['profit_rate'])
# np.mean(profit2017['profit_rate'])


# In[230]:


len(IF['收盘价(元)'].values[8764:20864])


# In[106]:


# nn = np.concatenate([[0],neg_profit_rate])
# nn = neg_profit_rate
plt.plot(profit2017['sell_time'],(profit2017['profit_rate']+1).cumprod()*IF['收盘价(元)'].values[9864]
         ,'b',IF['收盘价(元)'].values[9864:21964],'r')
plt.legend(['净值增长','标的价格'])
plt.title('2017IC: 净值增长 vs 标的价格')


# In[ ]:


#收益率计算
#2018年
#IF 0.05
# data = IF_close[21964:]
#IC 0.22
index =(IF['日期'].dt.year == 2018)
values = IF['收盘价(元)'].values[index]
data = IF_close[20064:]
#IH 0.089
# data = IF_close[20064:]
profit_rate = []
sell = []
sell_time = []
buy = []
buy_time = []
acc_profit2018 = 0.0

neg_buy = []
neg_sell = []
neg_profit_rate = []

MA10 = np.zeros(len(data))
freq_list = np.empty(len(data))
cyc_list = np.empty(len(data))
n = 40
k = 2
f = np.linspace(0.01,0.99,200)

for i in range(n,len(data)):
#     k,a = find_k(data[i-n:i])
    a = burgs(data[i-n:i],k)
#     fpe = FPE(data[i-n+1:i+1],a)
    sigma = sigma_2(data[i-n:i],k)[0]
#     sigma = np.std(a)
    y = map(lambda x: spectral_density(sigma,a,x),f)
    y = list(y)
    freq_list[i] = f[np.where(y==max(y))][0]
    cyc_list[i] = 1/freq_list[i]
    if(i >= n+10):
        MA10[i] = min(max(np.mean(cyc_list[i-10:i]),5),30)
        if((MA10[i]>=10) & (MA10[i-1]<10)):#上穿买入
#         if((cyc_list[i]>=10) & (cyc_list[i-1]<10)):
#             print('buy')
            buy_ = IF['收盘价(元)'].values[20064+i-1]
            buy_time_ = i
            if(len(sell)>0):
                buy.append(-sell_)
                buy_time.append(sell_time_)
                sell.append(-buy_)
                sell_time.append(buy_time_)
                profit_rate.append(sell_/buy_-1)
                acc_profit2018 = acc_profit2018 + sell_- buy_
        if((MA10[i]<=10)& (MA10[i-1]>10)):#下穿卖出
#         if((cyc_list[i]<=10) & (cyc_list[i-1]>10)):
#             print('sell')
            sell_ = IF['收盘价(元)'].values[20064+i-1]
            sell_time_ = i
            buy.append(buy_)
            buy_time.append(buy_time_)            
            sell.append(sell_)
            sell_time.append(sell_time_)
            profit_rate.append(sell_/buy_-1)
            acc_profit2018 = acc_profit2018 + sell_- buy_
    if(i%500==0):
        print(i)


# In[208]:


profit2018 = pd.DataFrame({'profit_rate':profit_rate,'sell':sell,'sell_time':sell_time,'buy':buy,'buy_time':buy_time})
acc_profit2018,np.sum(profit2018['profit_rate']),len(profit2018) 
#2018累计收益点数,累计收益率,交易次数


# In[43]:


aa = (profit2016['profit_rate']+1).cumprod()
(max(aa)-min(aa))*IF['收盘价(元)'].values[8764]


# In[62]:


np.sum(profit2017['profit_rate']<0)


# In[65]:


hh = profit2018.iloc[np.where(profit2018['profit_rate']<0)[0]]
np.mean(hh['profit_rate'])
# np.mean(profit2018['profit_rate'])


# In[109]:


# nn = np.concatenate([[0],neg_profit_rate])
# nn = neg_profit_rate
plt.plot(profit2018['sell_time'],(profit2018['profit_rate']+1).cumprod()*IF['收盘价(元)'].values[21964]
         ,'b',IF['收盘价(元)'].values[21964:],'r')
plt.rcParams['figure.figsize'] = (8, 4) 
plt.legend(['净值增长','标的价格'])
plt.title('2018IC: 净值增长 vs 标的价格')


# In[36]:


plt.plot(IF['收盘价(元)'].values)


# In[111]:


#收益率计算
#2016-18年
#IF 0.05
data = IF_close[0:]
#IC 0.22
# data = IF_close[20864:]
#IH 0.089
# data = IF_close[20064:]
profit_rate = []
sell = []
sell_time = []
buy = []
buy_time = []
diff = []
acc_profit = 0.0

neg_buy = []
neg_sell = []
neg_profit_rate = []

MA10 = np.zeros(len(data))
freq_list = np.empty(len(data))
cyc_list = np.empty(len(data))
n = 40
k = 2
f = np.linspace(0.01,0.99,200)

for i in range(n,len(data)):
#     k,a = find_k(data[i-n:i])
    a = burgs(data[i-n:i],k)
#     fpe = FPE(data[i-n+1:i+1],a)
    sigma = sigma_2(data[i-n:i],k)[0]
#     sigma = np.std(a)
    y = map(lambda x: spectral_density(np.sqrt(sigma),a,x),f)
    y = list(y)
    freq_list[i] = f[np.where(y==max(y))][0]
    cyc_list[i] = 1/freq_list[i]
    if(i >= n+10):
        MA10[i] = min(max(np.mean(cyc_list[i-10:i]),5),30)
        if((MA10[i]>=10) & (MA10[i-1]<10)):#上穿买入
#         if((cyc_list[i]>=10) & (cyc_list[i-1]<10)):
#             print('buy')
            buy_ = IF['收盘价(元)'].values[i-1]
            buy_time_ = i
            if(len(sell)>0):
                buy.append(-sell_)
                buy_time.append(sell_time_)
                sell.append(-buy_)
                sell_time.append(buy_time_)
                profit_rate.append(sell_/buy_-1)
                acc_profit = acc_profit + sell_- buy_
        if((MA10[i]<=10)& (MA10[i-1]>10)):#下穿卖出
#         if((cyc_list[i]<=10) & (cyc_list[i-1]>10)):
#             print('sell')
            sell_ = IF['收盘价(元)'].values[i-1]
            sell_time_ = i
            buy.append(buy_)
            buy_time.append(buy_time_)            
            sell.append(sell_)
            sell_time.append(sell_time_)
            profit_rate.append(sell_/buy_-1)
            acc_profit = acc_profit + sell_- buy_
    if(i%500==0):
        print(i)


# In[112]:


profit = pd.DataFrame({'profit_rate':profit_rate,'sell':sell,'sell_time':sell_time,'buy':buy,'buy_time':buy_time})
acc_profit,np.sum(profit['profit_rate']),len(profit) 
#2016-2018累计收益点数,累计收益率,交易次数


# In[73]:


np.sum(profit['profit_rate']<0)


# In[76]:


hh = profit.iloc[np.where(profit['profit_rate']<0)[0]]
np.mean(hh['profit_rate'])
# np.mean(profit2017['profit_rate'])


# In[114]:


# nn = np.concatenate([[0],neg_profit_rate])
# nn = neg_profit_rate
plt.plot(profit['sell_time'],(profit['profit_rate']+1).cumprod()*IF['收盘价(元)'].values[0]
         ,'b',IF['收盘价(元)'].values[0:],'r')
plt.rcParams['figure.figsize'] = (20, 4) 
plt.legend(['净值增长','标的价格'])
plt.title('2016-2018IC: 净值增长 vs 标的价格')

