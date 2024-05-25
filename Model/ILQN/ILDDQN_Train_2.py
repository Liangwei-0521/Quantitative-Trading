# ILDDQN Imitative Learning
import datetime
import matplotlib.pyplot as plt
from parameters import args_parameters
import torch
from Bad.Data.stock_data import data
from Model.ILQN.IL_Agent import IL_Agent_2
from Env.trainEnv import portfolio_trainEnv
# 专家训练
from Expert.Imitate_Expert import Expert


def IL_DDQN_train(episode, minimum, stockName, args):
    link = r'..\..\Data\\' + stockName + '.csv'
    train_df = data(link, window_length=args.window_length, t=2000).train_data()
    # 训练环境
    Env = portfolio_trainEnv(day=0, balance=100000, stock=train_df, cost=args.cost)
    # 专家
    expert = Expert(path=link, timeperiod=args.timeperiod)
    expert_transition_dict = expert.sample_expert_data(K=2000)
    # 智能体
    agent = IL_Agent_2(state_dim=args.state_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim,
                       lr=0.001,
                       device="cuda:0", gamma=0.95, epsilon=0.01,
                       target_update=10)

    return_List = []
    for i in range(episode):
        # print('Episode:', i)
        done = False
        episode_return = 0
        state = Env.reset()  # 返回初始状态
        state = torch.tensor(state.values, dtype=torch.float32).view(1, -1).to(device="cuda:0")
        # 经验池
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }
        start = datetime.datetime.now()
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = Env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action + 1)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            next_state = torch.tensor(next_state.values, dtype=torch.float32).view(1, -1).to(device="cuda:0")
            transition_dict['next_states'].append(next_state)
            state = next_state
            episode_return += reward
            if len(transition_dict['states']) >= minimum:
                # 算法优化 超参数灵敏度分析
                agent.update(transition_dict, expert_transition_dict, args.lambda_1, args.lambda_2)
        return_List.append(episode_return)
        end = datetime.datetime.now()
        print('运行时间：', end - start)

    # 模型保存
    PATH = r'..\..\Model\ILQN\result\DQN\agent_ddqn_' + stockName + '_' + str(args.lambda_1).replace('.', '_') + '.pt'
    # r'..\..\Result\cost\agent_ddqn_' + stockName + '_' + str(args.lambda_1).replace('.', '_') + '_' + '.pt'
    # r'..\..\Result\cost\agent_ddqn_' + stockName + '_' + str(args.cost).replace('.', '_') + '.pt'
    # r'..\..\Result\parameter_sensitivity\agent_ddqn_' + stockName + '_' + str(args.lambda_1).replace('.', '_') + '_' + '.pt'
    # r'..\..\Result\cost\agent_ddqn_' + stockName + '_' + str(args.lambda_1).replace('.', '_') + '_' + '.pt'
    torch.save(agent.state_dict(), PATH, _use_new_zipfile_serialization=False)

    # 每慕内奖励总和
    time = []
    for i in train_df:
        time.append(str(i.index.unique().values[-1]))
    Name = link.split('\\')[-1].split('.')[0]
    # 策略收益率
    plt.figure(figsize=(10, 3))
    plt.title('ILDQN: The Episode Reward of ' + Name)
    plt.grid()
    plt.plot(range(1, len(return_List) + 1), return_List)
    plt.show()
    # return return_List


if __name__ == '__main__':
    # Lambda 参数 灵敏度分析
    # 模型参数
    import os

    stockName = [i.split('.')[0] for i in os.listdir(r'../../Bad/Data')][:-2]

    args = args_parameters()
    # IL_DDQN_train(episode=25, minimum=1500, stockName=stockName[0], args=args)

    # 批量训练: 参数灵敏度
    print('Cost', args.cost)
    print('window', args.window_length)
    for name in stockName:
        print('Stock', name)
        IL_DDQN_train(episode=1, minimum=1500, stockName=name, args=args)

    # print('Stock', stockName[4])
    # IL_DDQN_train(episode=2, minimum=1500, stockName=stockName[4], args=args)
