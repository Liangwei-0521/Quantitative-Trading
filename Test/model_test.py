import math
from datetime import datetime
import numpy as np
import torch
from Data.stock_data import data
from Env.tradeEnv import portfolio_tradeEnv
from Model.BC.Behavioral_Cloning_Agent import BehaviorClone
from Model.DDQN.Dueling_DQN import DDQN_Agent
from Model.DQN.Deep_Q_Network import DQN_Agent
from Model.ILQN.IL_Agent import IL_Agent
from Model.ILQN.IL_Agent import IL_Agent_2
from parameters import args_parameters


def sharpe_Ratio(name, rate):
    # 无风险收益率：3%
    # 年化收益率：
    APY = (100000 * (rate[-1] - 1) / 100000 / 504) * 365
    print(name, 'APY:', APY)
    # sharpe_ratio = (APY - 0.03) / np.std(rate)
    Rate = []
    for i in range(len(rate) - 1):
        for j in range(1, len(rate)):
            Rate.append((rate[j] - rate[i]) / rate[i])
    # 日收益率转为年化收益率
    sharpe_ratio = (APY - 0.03) / (np.std(Rate) * math.sqrt(2))
    return sharpe_ratio


#  最大回测率
def Maximum_Drawdown(return_list):
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    return (return_list[j] - return_list[i]) / (return_list[j])


class backTest:

    def __init__(self, name):
        self.name = name
        self.path = r'..\Data\\' + self.name + '.csv'
        self.data = data(self.path, window_length=15, t=2000).trade_data()

    def multi_test_BC(self, code, args):
        BC_agent = BehaviorClone(state_dim=150, hidden_dim=50, action_dim=3, device="cuda:0", lr=0.01,
                                 timeperiod=15,
                                 K=2000,
                                 path=r'..\Data\\' + code + '.csv',
                                 batch_size=512)

        trade_df = self.data

        # 测试Dueling DQN网络
        trade_Env = portfolio_tradeEnv(day=0, stock=trade_df, balance=100000, cost=args.cost)
        BC_agent.ddqn.load_state_dict(torch.load(r'..\Model\BC\Dueling_Q_network\\' + code + '_BC_DdQN.pt'))
        BC_agent.test_BC(env=trade_Env, agent=BC_agent.ddqn, trade_df=trade_df, type='BC DDQN')
        print('BC累计收益率：', code, trade_Env.rate[-1])
        return trade_Env.rate

    def DDQN_Test(self, stockName, args):
        trade_df = self.data
        # data(link=BC_agent.path, window_length=15, t=2000).trade_data()
        # 实例化智能体
        DDQN_agent = DDQN_Agent(state_dim=150, hidden_dim=30, action_dim=3, lr=0.001, device="cuda:0", gamma=0.95,
                                epsilon=0.01, target_update=10)
        env = portfolio_tradeEnv(day=0, balance=100000, stock=trade_df, cost=args.cost)
        DDQN_agent.load_state_dict(torch.load(r'..\Model\DDQN\result\agent_ddqn_' + stockName + '.pt'))
        state = env.reset()  # 返回初始状态
        for i in trade_df:
            state = torch.tensor(state.values, dtype=torch.float32).view(1, -1).to(device="cuda:0")
            action = DDQN_agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
        print('DDQN累计收益率：', stockName, env.rate[-1])
        return env.rate

    def DQN_Test(self, stockName, args):
        trade_df = self.data
        # 实例化智能体
        DQN_agent = DQN_Agent(state_dim=150, hidden_dim=30, action_dim=3, lr=0.001, device="cuda:0", gamma=0.95,
                              epsilon=0.01, target_update=10)
        env = portfolio_tradeEnv(day=0, balance=100000, stock=trade_df, cost=args.cost)
        DQN_agent.load_state_dict(torch.load(r'..\Model\DQN\result\agent_dqn_' + stockName + '.pt'))
        state = env.reset()  # 返回初始状态
        for i in trade_df:
            state = torch.tensor(state.values, dtype=torch.float32).view(1, -1).to(device="cuda:0")
            action = DQN_agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
        print('DQN累计收益率：', stockName, env.rate[-1])
        return env.rate

    def ILDDQN_Test(self, stockName, lambda_1, args):
        trade_df = self.data
        # 测试环境
        env = portfolio_tradeEnv(day=0, balance=100000, stock=trade_df, cost=args.cost)
        agent = IL_Agent(state_dim=150, hidden_dim=50, action_dim=3, lr=0.001, device="cuda:0", gamma=0.95,
                         epsilon=0.01,
                         target_update=10)
        agent.load_state_dict(
            torch.load(
                r'..\Model\ILQN\result\DDQN\agent_ddqn_' + stockName + '_' + str(lambda_1).replace('.', '_') + '.pt'))
        # torch.load(r'..\Result\cost\agent_ddqn_' + stockName + '_' + str(lambda_1).replace('.', '_') + '_' + '.pt'))
        state = env.reset()  # 返回初始状态
        for i in trade_df:
            state = torch.tensor(state.values, dtype=torch.float32).view(1, -1).to(device="cuda:0")
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
        print('ILDDQN累计收益率：', stockName, env.rate[-1])
        return env.rate

    def ILDQN_Test(self, stockName, lambda_1, args):
        trade_df = self.data
        # data(link=BC_agent.path, window_length=15, t=2000).trade_data()
        # 测试环境
        env = portfolio_tradeEnv(day=0, balance=100000, stock=trade_df, cost=args.cost)
        agent = IL_Agent_2(state_dim=150, hidden_dim=50, action_dim=3, lr=0.001, device="cuda:0", gamma=0.95,
                           epsilon=0.01,
                           target_update=10)
        agent.load_state_dict(torch.load(
            r'..\Model\ILQN\result\DQN\agent_ddqn_' + stockName + '_' + str(lambda_1).replace('.', '_') + '.pt'))
        state = env.reset()  # 返回初始状态
        for i in trade_df:
            state = torch.tensor(state.values, dtype=torch.float32).view(1, -1).to(device="cuda:0")
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
        print('ILDQN 累计收益率：', stockName, env.rate[-1])
        return env.rate


if __name__ == '__main__':
    import os
    from parameters import args_parameters

    args = args_parameters()
    stockName = [i.split('.')[0] for i in os.listdir(r'../Data')][:-2]
    args = args_parameters()

    def Test(code):
        print(code)
        model_test = backTest(code)
        BC_rate = model_test.multi_test_BC(code, args)
        DQN_rate = model_test.DQN_Test(code, args)
        DDQN_rate = model_test.DDQN_Test(code, args)
        ILDDQN_rate = model_test.ILDDQN_Test(code, lambda_1=args.lambda_1, args=args)
        ILDQN_rate = model_test.ILDQN_Test(code, lambda_1=args.lambda_1, args=args)

        time = []
        for i in model_test.data:
            time.append(datetime.strptime(str(i.index.unique().values[-1]), '%Y%m%d').strftime('%Y-%m-%d'))
        name = model_test.path.split('\\')[-1].split('.')[0]

        # 平均投资
        balance = 100000
        price = model_test.data[0].Close.values[-1]
        shares = (1 - 0.003) * balance / price
        Market_rate = []
        for i in range(len(model_test.data)):
            Market_rate.append((shares * model_test.data[i].Close.values[-1] - balance) / balance + 1)

        import matplotlib.pyplot as plt
        from pylab import mpl
        mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体：解决plot不能显示中文问题
        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

        plt.rcParams['font.family'] = ['Times New Roman']
        plt.rcParams['xtick.direction'] = 'in'  # 刻度在里面
        plt.rcParams['ytick.direction'] = 'in'

        # 策略收益率
        plt.figure(figsize=(8.5, 4))
        # plt.title('The Cumulative Wealth Yield of ' + name)
        plt.grid()
        plt.xlabel('时间', fontdict={'family': 'SimSun', 'size': 10})
        plt.ylabel('累计收益率', fontdict={'family': 'SimSun', 'size': 10})

        plt.xticks(range(0, len(time), 70), )
        plt.plot(time[1:], Market_rate[1:], linewidth=1.5, label='Market', c='orange')
        plt.plot(time[1:], BC_rate, linewidth=1.5, label='BC', c='blue')
        plt.plot(time[1:], DQN_rate, linewidth=1.5, label='DQN', c='green')
        plt.plot(time[1:], DDQN_rate, linewidth=1.5, label='DDQN', c='purple')
        plt.plot(time[1:], ILDDQN_rate, linewidth=1.5, label='BCDDQN', c='r')
        # plt.plot(time[1:], ILDQN_rate, label='ILDQN')
        plt.legend()
        plt.savefig(r'E:\Code_Project\Agent with imitation\Pictures\\' + code + '1.svg', format='svg', dpi=150)
        plt.show()
        # ILDDQN策略指标
        IL_SP = sharpe_Ratio('ILDDQN', ILDDQN_rate)
        IL_MDD = Maximum_Drawdown(ILDDQN_rate)
        # 行为克隆策略指标
        BC_SP = sharpe_Ratio('BC', BC_rate)
        BC_MDD = Maximum_Drawdown(BC_rate)
        # 对决DQN策略指标
        DDQN_SP = sharpe_Ratio('DDQN', DDQN_rate)
        DDQN_MDD = Maximum_Drawdown(DDQN_rate)
        # DQN策略指标
        DQN_SP = sharpe_Ratio('DQN', DQN_rate)
        DQN_MDD = Maximum_Drawdown(DQN_rate)
        # 市场策略指标
        Market_SP = sharpe_Ratio('Market', Market_rate)
        Market_MDD = Maximum_Drawdown(Market_rate)
        # 夏普比率
        print('夏普比率:', 'ILDDQN:', IL_SP, 'BC:', BC_SP, 'DDQN:', DDQN_SP, 'DQN:', DQN_SP, 'Market:', Market_SP)
        # 最大回撤率
        print('最大回撤率:', 'ILDDQN:', IL_MDD, 'BC:', BC_MDD, 'DDQN:', DDQN_MDD, 'DQN:', DQN_MDD,  'Market:', Market_MDD)

    # for name in stockName:
    Test(stockName[2])

    # for name in stockName:
    #     Test(name)
