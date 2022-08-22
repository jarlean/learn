import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import scipy.stats

# 图形中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 随机漫步
def random_walk(walk_steps):
    k = 0
    y = []
    draws = np.random.randint(-1,2,size=walk_steps)
    # print(type(draws))
    for i in draws:
        k=k+i
        y.append(k)

    m = np.array(y)
    return m

def draw_random_walk():
    walk_steps=2000
    walk_path = random_walk(walk_steps)
    start_y=0
    start_x=0
    end_y=walk_path[-1]
    end_x = walk_steps - 1

    max_y=walk_path.max()
    max_x=walk_path.argmax()

    min_y=walk_path.min()
    min_x=walk_path.argmin()

    x = np.linspace(0,walk_steps,num=walk_steps)

    plt.plot(x,walk_path,label='walk step')

    # 添加标注
    plt.annotate(
        'start:({},{})'.format(start_x,start_y),
        xy=(start_x,start_y),
        xycoords='data',
        xytext=(+50,+20),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.5',fc='yellow',alpha=0.5),
        arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')
    )

    plt.annotate(
        'end:({},{})'.format(end_x,end_y),
        xy=(end_x,end_y),
        xycoords='data',
        xytext=(-50,-20),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.5',fc='yellow',alpha=0.5),
        arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')
    )

    plt.annotate(
        'max:({},{})'.format(max_x,max_y),
        xy=(max_x,max_y),
        xycoords='data',
        xytext=(-20,+20),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.5',fc='yellow',alpha=0.5),
        arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')
    )

    plt.annotate(
        'min:({},{})'.format(min_x,min_y),
        xy=(min_x,min_y),
        xycoords='data',
        xytext=(-20,+20),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.5',fc='yellow',alpha=0.5),
        arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')
    )

def simplot_random_walk():
    _ = [plt.plot(np.arange(2000),random_walk(2000),c='b',alpha=0.5) for _ in np.arange(0,1000)]

def sim_normal_distribution():
    end_path=[random_walk(walk_steps=2000)[-1] for _ in np.arange(0,1000)]
    _,bins,_ = plt.hist(end_path,bins=50,density=True)

plt.legend(loc='best')
plt.xlabel('漫走步数')
plt.ylabel('分布轨迹')
plt.title(u'模拟随机漫步')
# draw_random_walk()
# simplot_random_walk()
sim_normal_distribution()
plt.show()




'''
# 正态分布
# plt.hist(np.random.normal(loc=-2,scale=0.5,size=10000),bins=500,density=True,color='g')
# plt.hist(np.random.normal(loc=0,scale=1,size=10000),bins=500,density=True,color='b')
# plt.hist(np.random.normal(loc=2,scale=1.5,size=10000),bins=500,density=True,color='r')
# plt.show()

# plt.hist(np.random.normal(loc=0,scale=1,size=10000)*0.5-2,bins=50,density=True,color='g')
# plt.hist(np.random.normal(loc=0,scale=1,size=10000),bins=50,density=True,color='b')
# plt.hist(np.random.normal(loc=0,scale=1,size=10000)*1.5+2,bins=50,density=True,color='r')
# plt.show()

# print(np.sqrt(2)) # 根号2
# print(np.exp(2)) # e的二次方，e=2.718281828459045，e自然常数
a,bins,b = plt.hist(np.random.normal(loc=0,scale=1,size=10000),bins=50,density=True)
# plt.plot(bins,1. / (np.sqrt(2*np.pi)*1) * np.exp(-(bins-0)**2 / (2*1**2)) ,label='$\mu$=%.1f,$\sigma^2$=%.1f'%(0,1) ,lw=2)
plt.plot(bins,scipy.stats.norm.pdf(bins,loc=0,scale=1),label='$\mu$=%.1f,$\sigma^2$=%.1f'%(0,1) ,lw=2)
print("a",a)
print("bins",bins)
print("b",b)
plt.legend()
plt.show()
'''

'''
# 仓位管理
def positmanage(play_cnt=1000,stock_num=9,commission=0.01):
    my_money=np.zeros(play_cnt)
    my_money[0]=1000
    win_rate = random.uniform(0.5,1)
    binomial = np.random.binomial(stock_num,win_rate,play_cnt)
    for i in range(1,play_cnt):
        once_chip = round(my_money[i-1]*((win_rate*1-(1-win_rate))/1),2)  #投注金额
        if binomial[i-1] > stock_num//2:
            # print("win_rate:%s  my_money[%s]+once_chip,%s+%s=%s" %(win_rate, i-1,my_money[i-1],once_chip,my_money[i-1]+once_chip))
            my_money[i] = my_money[i-1]+once_chip
        else:
            # print("win_rate:%s  my_money[%s]-once_chip,%s-%s=%s" %(win_rate, i-1,my_money[i-1],once_chip,my_money[i-1]-once_chip))
            my_money[i] = my_money[i-1]-once_chip
        my_money[i]=my_money[i]-commission
        if my_money[i]<=0:
            break
    return my_money

plt.grid('--',color='blue',alpha=0.2)
trader = 50
# _ = [plt.plot(np.arange(4),positmanage(play_cnt=4,stock_num=9,commission=0.01)) for _ in np.arange(0,trader)]


print(positmanage(play_cnt=4,stock_num=9,commission=0.01))

_,a,b = plt.hist([positmanage(play_cnt=5)[-1] for _ in np.arange(0,trader)],bins=30)

print("_",_)
print("a",a)
print("b",b)

plt.show()
'''

'''
# 简单市场模型的博弈
# 胜率win_rate,次数play_cnt,股票数量stock_num,仓位比例position,手续费commission,加注标志lever
def simpmarket(win_rate,play_cnt=1000,stock_num=9,position=0.01,commission=0.01,lever=False):
    my_money=np.zeros(play_cnt)
    my_money[0] = 10000
    lose_cnt = 1
    binomial = np.random.binomial(stock_num,win_rate,play_cnt)
    # print("binomial",binomial)
    for i in range(1,play_cnt):
        if my_money[i-1]*position*lose_cnt <= my_money[i-1]: # 资金充足
            once_chip = my_money[i-1]*position
        else:
            print("loser",my_money[i-1])
            break
        if binomial[i] > stock_num //2: # 一半股票上涨
            my_money[i] = my_money[i-1]+once_chip if lever == False else my_money[i-1] + once_chip*lose_cnt
            lose_cnt = 1
        else:
            my_money[i] = my_money[i-1]-once_chip if lever == False else my_money[i-1] - once_chip*lose_cnt
            lose_cnt += 1
        my_money[i] -= commission
        if my_money[i] <= 0:
            print("empty",my_money[i])
            break
    return my_money

trader = 60
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.grid('-.')
ax2.grid('-.',alpha=0.2,color='red')
ax2.set_yticks(np.arange(0,40,2))

_ = [ax1.plot(np.arange(1000),simpmarket(0.5,play_cnt=1000,stock_num=9,commission=0.1,lever=True)) for _ in np.arange(0,trader)]
_ = ax2.hist([simpmarket(0.5,play_cnt=5000,stock_num=9,commission=0.1,lever=True)[-1] for _ in np.arange(0,trader)],bins=30)

plt.show()
'''


'''
# 伯努利分布
# 抛掷5次硬币，出现抛掷5次都为正面朝上的概率
res = sum(np.random.binomial(5,0.5,1000000)==5)/1000000
print(res) # 0.03125
'''

'''
# 对象式画图
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)

x = np.linspace(0,10,100)
y = np.sin(x)
y2 = np.cos(x)
ax.plot(x,y,'--g',label='sin(x)',lw=2)
ax2.plot(x,y2,'--r',label='cos(x)',lw=2)

ax.set_title('对象式画图sin')
ax2.set_title('对象式画图cos')
# ax.set_xlimit()

ax.legend(loc='upper right',fontsize=13)
ax2.legend(loc='upper right',fontsize=13)

ax.set_xlim(0,8)
ax.set_ylim(-1.5,1.5)

ax.set_xticks(np.arange(1,10,2),)
ax.set_yticks(np.arange(-1,1.5,1),)

ax.set_xticklabels(['2022-01-01','2022-02-01','2022-03-01','2022-04-01','2022-05-01'],rotation=45)
ax.set_yticklabels(['最小值','零值','最大值'])

ax.set_xlabel('x轴')
ax.set_ylabel('y轴')

ax.grid(ls=':',color='red')

# ax.set_xlabel
#
# plt.xlabel
# plt.xlim()

plt.show()
'''


'''
# 函数式画图
# 字体设置
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12,8))
x=np.linspace(0,10,100)
y=np.sin(x)
print(x)
print(y)
plt.plot(x,y,'--g',label='sin(x)',lw=2)
plt.xlabel('x轴')
plt.ylabel('y轴')

plt.xticks(np.arange(1,10,2),('2022-01-01','2022-02-01','2022-03-01','2022-04-01','2022-05-01'),rotation=45)
plt.yticks(np.arange(-1,1.5,1),('最小值','零值','最大值'))
plt.xlim(0,10)
plt.ylim(-1.5,1.5)

plt.grid(True,ls=':',color='red',alpha=0.5)

plt.title('sin(x)的函数式画法')

plt.legend(loc = 'upper right')

plt.show()
'''

'''
## 构建股票行情数据
close_data = np.random.normal(loc=100.0, scale=10, size=1000)
# print("close_data",close_data)
open_data = np.roll(close_data,1)
# print("open_data",open_data)
high_data = np.where(open_data>=close_data,open_data,close_data)
# print("high_data",high_data)
low_data = np.where(open_data<=close_data,open_data,close_data)
# print("low_data",low_data)
array_ind = pd.date_range(start='2021-01-01', periods=1000, freq='D')
# print("array_ind",array_ind)
array_col = ['close']
# print("array_ind",array_col)
# 第一个交易日开盘价应为无效
open_data[0],close_data[0],high_data[0],low_data[0]=np.nan,np.nan,np.nan,np.nan,
array_data = pd.DataFrame(data={"open":open_data,"close":close_data,
                                "high":high_data,"low":low_data}
                          , index=array_ind)
# print("array_data",array_data)
# print(array_data.tail(10))
# print(array_data.shape)
# print(array_data.describe(include='all'))
# print(array_data.info())

# 交易量为4000的正态分布
array_data_volume = pd.DataFrame({'volume':np.round(np.random.normal(loc=4000,scale=1000,size=1000),0)},index=array_ind)
# print(array_data_volume)

# 删除空值行，并替换
array_data.dropna(axis=0,how='any',inplace=True)
# print(array_data)

# 使用pandas.concat()按列方向合并，各轴索引的交集
array_concat = pd.concat([array_data,array_data_volume],axis=1,join='inner')
# print(array_concat.head())

# 使用pandas.merge()横向合并
array_concat = pd.merge(array_data, array_data_volume, left_index=True, right_index=True, how='inner')
# print(array_concat.head())

# 使用pandas.join()横向合并
array_concat = array_data.join(array_data_volume,how='inner')
# print(array_concat.head())

# for-in循环生成振幅
# array_data = array_data.assign(pct_change = 0)
# for i in np.arange(0,array_data.shape[0]):
# array_data.columns.get_loc('pct_change')取pct_change所在列的序号
#     array_data.iloc[i,array_data.columns.get_loc('pct_change')] \
#         = (array_data.iloc[i]['high'] - array_data.iloc[i]['low'])/array_data.iloc[i]['open']
#
# print(array_data[0:10])

# iterrows()循环生成振幅
# array_data = array_data.assign(pct_change = 0)
# for index,row in array_data.iterrows():
#     array_data.loc[index,'pct_change'] = (row['high'] - row['low']) / row['open']
#
# print(array_data.head())

# applay()循环生成振幅
# array_data['pct_change'] = array_data.apply(lambda row:(row['high']-row['low'])/row['open'],axis=1)
# print(array_data)

# pandas series矢量生成振幅
# array_data['pct_change'] = (array_data['high'] - array_data['low'])/array_data['open']
# print(array_data.head())

# numpy array矢量生成振幅
array_data['pct_change'] = (array_data['high'].values - array_data['low'].values)/array_data['open'].values
print(array_data.head())

# 保留2位精度
# array_data = array_data.round(2)
# print(array_data.info())

# 找出空值的行记录
# print(array_data[array_data.isnull().T.any().T])
'''


'''
## 可视化
df_visual = array_data.loc['2021-01-01':'2022-01-01',['high','low']].plot(linewidth=1,figsize=(8,6))
df_visual.set_xlabel('Time')
df_visual.set_ylabel('High and Low price')
df_visual.set_title('From 2021-01-01 To 2022-01-01')
df_visual.legend()
plt.show()
'''

'''
## 重采样resample
rng = pd.date_range('20220101',periods=12,freq='D')
ts_d = pd.Series(np.arange(1,13),index=rng)
print(ts_d)

print(ts_d.resample('5D',closed='left',label='left').sum())
print(ts_d.resample('5D',closed='right',label='right').sum())
print(ts_d.resample('5D',closed='right',label='left').sum())
'''

'''
## 矩阵array
array_x = np.array([ [1,1,1,3331,1], [1111,2,2,2,2111] ])
print(array_x.strides)

array_randint = np.random.randint(1,4,size=(5,2,10))
print(array_randint)
'''

'''
## 时间运算Timedelta
pddatetime1 = pd.to_datetime(datetime(2022,8,10,21,22,2))
print(type(pddatetime1), pddatetime1)

pddatetime2 = pd.to_datetime('2022-8-10 20:22:2')
print(type(pddatetime2), pddatetime2)

dt = datetime(2022, 8, 10, 22, 22, 2)
print(type(dt), dt)

print(dt - pddatetime2)
print(pddatetime1 - pddatetime2)
print(pddatetime1 + pd.Timedelta(days=5,minutes=20))
print(dt + pd.Timedelta(days=5,minutes=20))
'''