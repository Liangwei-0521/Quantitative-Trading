import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class Expert:

    def __init__(self, path, timeperiod) -> None:
        self.path = path
        self.timeperiod = timeperiod
        self.data = pd.read_csv(path).set_index('Date')

    def Expert_strategy(self, data):
        # 实现最高点卖出，最低点买入专家策略
        data['Action'] = np.zeros(shape=len(data), dtype=int)
        for i in range(len(data.loc[:, 'Close'])):
            if data.Open.values[i] < data.Close.values[i]:  # 上涨行情
                data.iloc[i, -1] = 1
            if data.Open.values[i] > data.Close.values[i]:  # 下跌行情
                data.iloc[i, -1] = -1
            if data.Open.values[i] == data.Close.values[i]:  # 振荡行情
                data.iloc[i, -1] = 0
        data.to_csv('Expert_data.csv')
        return data

    def sample_expert_data(self, K):
        # 执行专家策略
        data = self.Expert_strategy(self.data)
        # 存储专家策略经验
        B = []
        # 针对训练数据
        df = data.loc[:, ['Close', 'Open', 'High', 'Low', 'RSI', 'ROC', 'CCI', 'MACD', 'EXPMA', 'VMACD', 'Action']]
        B.insert(-1, [df[i - self.timeperiod:i] for i in range(self.timeperiod, len(df))])
        Expert_transition_dict = {
            'states': [],
            'actions': [],
        }

        def ES_Sample(train_data):
            for j, i in enumerate(train_data[:-1]):
                Expert_transition_dict['states'].append(i.iloc[:, :-1].values.reshape(1, -1))
                Expert_transition_dict['actions'].append(
                    train_data[j + 1].Action.values[-1] + 1)  # 加1是因为智能体的动作取值是：[0, 1, 2], 环境内是[-1, 0, 1]
            return Expert_transition_dict

        ES_Sample(B[0][:K])
        return Expert_transition_dict


if __name__ == '__main__':
    # 专家策略的回测
    import matplotlib.pyplot as plt
    # from Env.tradeEnv import portfolio_tradeEnv
    from Expert.trade_Env import portfolio_tradeEnv
    from Bad.Data.stock_data import data

    E = Expert(path=r'../DF/000001_SZ.csv', timeperiod=15)

    DF = E.sample_expert_data(K=3000)
    print(len(DF['states']), len(DF['actions']))

    link = r'..\Data\000001_SZ.csv'

    name = link.split('\\')[-1].split('.')[0]
    trade_df = data(link, window_length=15, t=2000).train_data()
    env = portfolio_tradeEnv(day=0, stock=trade_df, balance=100000, cost=0.000)
    # 专家策略回测结果
    actions = DF['actions'][:2000]

    state = env.reset()
    for i in actions:
        next_state, reward, done, _ = env.step(i - 1)
        state = next_state

    time = []
    for i in trade_df:
        time.append(str(i.index.unique().values[-1]))

    # 平均投资
    balance = 100000
    price = trade_df[0].Close.values[-1]
    shares = (1 - 0.015) * balance / price
    Market_rate = []
    for i in range(len(trade_df)):
        Market_rate.append((shares * trade_df[i].Close.values[-1] - balance) / balance + 1)
    plt.figure(figsize=(10, 5))
    plt.title('The Expert Log Yield of ' + name)
    plt.grid()
    plt.xticks(range(0, len(time), 235), )
    plt.plot(time[1:], env.rate, label='Expert')
    plt.legend()
    plt.show()

    # 策略收益率
    plt.figure(figsize=(10, 5))
    plt.title('The Expert Cumulative wealth rate of ' + name)
    plt.grid()
    plt.xticks(range(0, len(time), 35), )
    plt.plot(time[1:][-250:], env.rate[-250:], label='Expert')

    import pandas as pd
    expert_rate = pd.Series(env.rate[-250:]).to_csv('expert.csv')
    # print(env.rate[-250:])
    plt.legend()
    plt.show()

    # 策略收益率
    plt.figure(figsize=(10, 3))
    plt.title('The Market Cumulative wealth of ' + name)
    plt.grid()
    plt.xticks(range(0, len(time), 230), )
    # plt.plot(time[1:], env.rate, label=type)
    plt.plot(time[1:], Market_rate[1:], c='#CD5C5C', label='Market')
    plt.legend()
    plt.show()





