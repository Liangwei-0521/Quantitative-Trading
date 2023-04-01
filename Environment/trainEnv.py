import numpy as np


class stock_trainEnv:
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
        self.len = len(self.stock)
        self.n_steps = 1

    def step(self, action):

        begin_asset_value = self.balance + self.stock_state.Close.values[-1] * self.shares[-1]
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
        self.n_steps += 1
        self.stock_state = self.stock[self.day]
        end_asset_value = self.balance + self.stock_state.Close.values[-1] * self.shares[-1]
        # 累计收益率
        self.rate.append((end_asset_value - 1) / 1)
        # 奖励函数
        self.reward = np.log(end_asset_value / begin_asset_value)
        return self.stock_state, self.reward, self.is_end(), {}

    def buy(self, action):
        # 账户的钱是否支持买入行为
        if self.balance > 0:
            self.shares.append((1 - self.transaction_cost) * self.balance / self.stock_state.Close.values[-1])
            self.balance = 0
            # print('Buy Share:', action * self.HMAX_SHARE)
        else:
            pass

    def hold(self, action):
        pass

    def sell(self, action):
        # 全部卖出
        cash = self.stock_state.Close.values[-1] * self.shares[-1] * (1 - self.transaction_cost)
        self.balance += cash
        # 更改份额
        self.shares.append(0)

    def is_end(self, ):
        if self.n_steps >= self.len - 1:
            print('Day:', self.day, '该幕的对数收益：', self.reward, '该幕的累计收益率：', self.rate[-1])
            self.terminal = True
        else:
            self.terminal = False
        return self.terminal

    def reset(self, ):
        self.n_steps = 0
        self.balance = 1
        self.rate = []
        self.shares = [0] * 1
        self.terminal = False
        self.day = 0  # np.random.randint(low=0, high=len(self.stock) - self.len - 1)
        self.stock_state = self.stock[self.day]
        return self.stock_state
