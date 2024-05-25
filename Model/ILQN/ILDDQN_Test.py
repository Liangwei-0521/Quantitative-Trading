from Model.ILQN.IL_Agent import IL_Agent
from Env.tradeEnv import portfolio_tradeEnv
from parameters import args_parameters
import torch
from Data.stock_data import data


def IL_DDQN_Test(stockName, args):
    # 训练数据
    link = r'..\..\Data\\' + stockName + '.csv'
    trade_df = data(link=link, window_length=args.window_length, t=2000).trade_data()
    # 测试环境
    env = portfolio_tradeEnv(day=0, balance=100000, stock=trade_df, cost=args.cost)
    agent = IL_Agent(state_dim=args.state_dim, hidden_dim=args.hidden_dim, action_dim=args.action_dim,
                     lr=0.001,
                     device="cuda:0", gamma=0.95, epsilon=0.01,
                     target_update=10)
    # 模型导入
    agent.load_state_dict(torch.load(r'..\..\Result\cost\agent_ddqn_' + stockName + '_' + str(args.lambda_1).replace('.', '_') + '_' + '.pt'))
    # r'..\..\Result\parameter_sensitivity\agent_ddqn_' + stockName + '_' + str(args.lambda_1).replace('.', '_') +'_' + '.pt'
    # r'..\..\Model\ILQN\result\DDQN\agent_ddqn_'+stockName+'_'+str(args.lambda_1).replace('.', '_')+'.pt'
    # r'..\..\Result\cost\agent_ddqn_' + stockName + '_' + str(args.lambda_1).replace('.', '_') + '_' + '.pt'
    # r'..\..\Result\parameter_sensitivity\agent_ddqn_' + stockName + '_' + str(args.lambda_1).replace('.', '_') +'_' + '.pt'
    state = env.reset()  # 返回初始状态
    for i in trade_df:
        state = torch.tensor(state.values, dtype=torch.float32).view(1, -1).to(device="cuda:0")
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
    print('ILDDQN累计收益率：', stockName, env.rate[-1])

    return env.rate


if __name__ == '__main__':
    # Lambda 参数 灵敏度分析
    # 模型参数
    import os
    stockName = [i.split('.')[0] for i in os.listdir(r'../../Data')][:-2]
    args = args_parameters()
    print('Cost:', args.cost)
    print('window', args.window_length)
    print('lambda:', args.lambda_1)
    # for name in stockName:
    #     print('Stock:', name)
    #     IL_DDQN_Test(stockName=name, args=args)
    IL_DDQN_Test(stockName=stockName[0], args=args)
