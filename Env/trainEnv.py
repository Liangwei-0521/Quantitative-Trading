import numpy as np


class portfolio_trainEnv:
    # action = [-1, 0, 1]

    def __init__(self, day, stock, balance, cost) -> None:
        self.day = day
        self.stock = stock  # 数据
        self.stock_state = self.stock[self.day]
        self.balance = balance
        self.shares = [0] * 1
        self.transaction_cost = cost
        self.terminal = False
        self.rate = []
        self.reward = 0
        self.reward_List = []

    def step(self, action):
        self.terminal = self.day >= len(self.stock) -1
        if self.terminal:
            # print('Balance:', self.balance, 'Close Price:', self.stock_state.Close.values[-1], 'Shares:', self.shares[-1],
            #       'Values:', self.balance+self.stock_state.Close.values[-1] * self.shares[-1])
            return self.stock_state, self.reward, self.terminal, {}

        else: 
            begin_assert_value = self.balance + self.stock_state.Close.values[-1] * self.shares[-1]
            if action == -1:
                # 执行卖出动作
                self.sell(action)
            if action == 0:
                # 执行持有动作，即不执行任何交易行为
                self.hold(action)
            if action == 1:
                # 执行买入动作
                self.buy(action)

            self.day += 1
            # print('Day:', self.day)
            self.stock_state = self.stock[self.day]
            end_assert_value = self.balance + self.stock_state.Close.values[-1] * self.shares[-1]
            # 累计收益率
            self.rate.append((end_assert_value - 100000) / 100000)
            # 奖励函数
            self.reward = np.log(end_assert_value/begin_assert_value)
            self.reward_List.append(self.reward)
            # print(self.day, 'Balance:', self.balance, 'Close Price:', self.stock_state.Close.values[-1], 'Shares', self.shares[-1], 'Values:', end_assert_value)
            return self.stock_state, self.reward, self.terminal, {}
            
    def buy(self, action):
        # 账户的钱是否支持买入行为
        if self.balance > 0:
            self.shares.append((1-self.transaction_cost)*self.balance/self.stock_state.Close.values[-1])
            self.balance = 0 
            # print('Buy Share:', action * self.HMAX_SHARE)
        else:
            pass
            
    def hold(self, action):
        pass

    def sell(self, action):
        # 全部卖出
        cash = self.stock_state.Close.values[-1]*self.shares[-1]*(1-self.transaction_cost)
        self.balance += cash
        # 更改份额
        self.shares.append(0)

    def reset(self, ):
        self.day = 0
        self.balance = 100000
        self.stock_state = self.stock[self.day]
        self.terminal = False
        return self.stock_state
        

