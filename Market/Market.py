# from Env.tradeEnv import portfolio_tradeEnv
from Expert.trade_Env import portfolio_tradeEnv
from Bad.Data.stock_data import data

link = r'..\Data\002415_SZ.csv'

name = link.split('\\')[-1].split('.')[0]
trade_df = data(link, window_length=15, t=2000).trade_data()
env = portfolio_tradeEnv(day=0, stock=trade_df, balance=100000, cost=0.000)
time = []

for i in trade_df:
    time.append(str(i.index.unique().values[-1]))

# 平均投资
balance = 100000
price = trade_df[0].Close.values[-1]
shares = (1 - 0.006) * balance / price
Market_rate = []
for i in range(len(trade_df)):
    Market_rate.append((shares * trade_df[i].Close.values[-1] - balance) / balance )

print(Market_rate[-1])