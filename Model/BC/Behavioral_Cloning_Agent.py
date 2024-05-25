# 专家模仿学习---行为克隆
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Expert.Imitate_Expert import Expert


class DQN_Policy(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(DQN_Policy, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim).to(device)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.linear4 = nn.Linear(hidden_dim, action_dim).to(device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        action = F.relu(self.linear4(F.relu(self.linear3(x))))
        return action


class DDQN_Policy(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(DDQN_Policy, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim).to(device)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc_A = nn.Linear(hidden_dim, action_dim).to(device)
        self.fc_V = nn.Linear(hidden_dim, 1).to(device)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        A = self.fc_A(F.relu(self.linear3(x)))
        V = self.fc_V(F.relu(self.linear4(x)))
        return A + V - A.mean(1).view(-1, 1)


class BehaviorClone:

    def __init__(self, state_dim, hidden_dim, action_dim, device, lr, path, timeperiod, K,
                 batch_size) -> None:
        super(BehaviorClone, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.path = path  # 专家数据集合
        self.K = K  # 划分训练集和测试集的长度
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        # 实例化专家
        self.expert = Expert(path=path, timeperiod=timeperiod)
        self.expert_transition_dict = self.expert.sample_expert_data(K=K)
        # 策略网络
        self.dqn = DQN_Policy(state_dim, hidden_dim, action_dim, device)
        self.ddqn = DDQN_Policy(state_dim, hidden_dim, action_dim, device)
        self.batch_size = batch_size

    def learn(self, episode, module, lr, path):
        criterion = nn.CrossEntropyLoss()
        for i in range(episode):
            index = np.random.randint(low=0, high=self.K - 2, size=self.batch_size)
            states = torch.tensor(np.array(self.expert_transition_dict['states'])[index]).to(self.device).squeeze(
                dim=1).to(torch.float)
            actions = torch.tensor(np.array(self.expert_transition_dict['actions'])[index], dtype=torch.int64).view(-1, 1).to(self.device)
            # 方法1：最大似然估计
            # log_probs = torch.log(module(states).gather(1, actions))
            # bc_loss = torch.mean(-log_probs)

            # 方法2：交叉熵损失函数
            states = module(states).float()
            actions = F.one_hot(actions, num_classes=3).squeeze(dim=1).to('cuda:0').float()
            bc_loss = criterion(states, actions)

            optimizer = optim.Adam(module.parameters(), lr=lr)
            optimizer.zero_grad()
            bc_loss.backward()
            optimizer.step()
            print('Episode:', i, 'The Imitate Agent has learned, the Loss is: {}'.format(bc_loss))
        torch.save(module.state_dict(), path, _use_new_zipfile_serialization=False)

    def train_BC(self, episode, dqn_path, ddqn_path, type):
        if type == 'DDQN':
            self.learn(episode, self.ddqn, self.lr, ddqn_path)
        else:
            dqn = DQN_Policy(self.state_dim, self.hidden_dim, self.action_dim, self.device)
            self.learn(episode, dqn, self.lr, dqn_path)

    def test_BC(self, env, agent, trade_df, type):
        #  智能体测试
        state = env.reset()  # 返回初始状态
        for i in range(len(trade_df)):
            state = torch.tensor(state.values, dtype=torch.float32).view(1, -1).to(device='cuda:0')
            # 采用模仿智能体的动作函数
            action = agent(state).argmax().item() - 1
            next_state, reward, done, _ = env.step(action)
            state = next_state
        # time = []
        # for i in trade_df:
        #     time.append(str(i.index.unique().values[-1]))
        # name = self.expert.path.split('\\')[-1].split('.')[0]
        #
        # # 平均投资
        # balance = 100000
        # price = trade_df[0].Close.values[-1]
        # shares = (1 - 0.015) * balance / price
        # Market_rate = []
        # for i in range(len(trade_df)):
        #     Market_rate.append((shares * trade_df[i].Close.values[-1] - balance) / balance + 1)
        #
        # # 策略收益率
        # plt.figure(figsize=(10, 3))
        # plt.title('The Cumulative wealth of ' + name)
        # plt.grid()
        # plt.xticks(range(0, len(time), 70), )
        # plt.plot(time[1:], env.rate, label=type)
        # plt.plot(time[1:], Market_rate[1:], c='#CD5C5C', label='Market')
        # plt.legend()
        # plt.show()


if __name__ == '__main__':
    # 实例化克隆

    bc_agent = BehaviorClone(state_dim=150, hidden_dim=50, action_dim=3, device="cuda:0", lr=0.01,
                             timeperiod=15,
                             K=2000,
                             path=r'../../DF\000001_SZ.csv',
                             batch_size=512)
    # 训练
    bc_agent.train_BC(episode=2500,
                      dqn_path=r'.\Q_network\bc_DQN.pt',
                      ddqn_path=r'.\Dueling_Q_network\bc_DDQN.pt',
                      type='DDQN')

    bc_agent.train_BC(episode=2500,
                      dqn_path=r'.\Q_network\bc_DQN.pt',
                      ddqn_path=r'.\Dueling_Q_network\bc_DDQN.pt',
                      type='DQN')

    # 测试
    from Data.stock_data import data
    from Env.tradeEnv import portfolio_tradeEnv

    # 行为克隆智能体
    bc_agent = BehaviorClone(state_dim=150, hidden_dim=50, action_dim=3, device="cuda:0", lr=0.01,
                             timeperiod=15,
                             K=2000,
                             path=r'../../DF\000001_SZ.csv',
                             batch_size=512)
    # 模拟交易数据
    trade_df = data(link=bc_agent.path, window_length=15, t=2000).trade_data()
    tradeEnv = portfolio_tradeEnv(day=0, stock=trade_df, balance=100000, cost=0.003)
    # 测试DQN网络
    bc_agent.dqn.load_state_dict(torch.load(r'.\Q_network\bc_DQN.pt'))
    bc_agent.test_BC(env=tradeEnv, agent=bc_agent.dqn, trade_df=trade_df)
    del tradeEnv

    # 测试Dueling DQN网络
    tradeEnv = portfolio_tradeEnv(day=0, stock=trade_df, balance=100000, cost=0.003)
    bc_agent.ddqn.load_state_dict(torch.load(r'.\Dueling_Q_network\bc_DDQN.pt'))
    bc_agent.test_BC(env=tradeEnv, agent=bc_agent.ddqn, trade_df=trade_df)
