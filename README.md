# Quantitative-Trading
Single stock trading strategies based on DQN, DDQN, Behavioural cloning and BCDDQN. 

## Technology Stack

- Pytorch
- Request

## Result

![项目截图](https://github.com/1998-Chen/Quantitative-Trading/blob/main/Result/000001_SZ.svg)

## Code examples
#### Trading Logic
        def make_action(self, action, begin_value):
        if self.shares[-1] == 0:
            # 空仓
            if action == 1:  # 买入
                if self.balance > 0:
                    # 全部买入
                    self.shares[-1] = (1 - self.transaction_cost) * self.balance / self.stock[self.day-1].close.values[-1]
                    self.balance = 0
                    self.reward = np.log(self.balance + self.stock[self.day].close.values[-1] * self.shares[-1] - begin_value)
            if action == 0:
                # 无风险利润
                self.reward = 0.0015
            if action == -1:
                self.reward = 0
        else:
            # 全仓
            if action == 1:
                self.reward = (self.stock[self.day-1].close[-1] / self.stock[self.day].close[-1])-1 
            if action == 0:
                # 持有：价格变化
                self.reward = (self.stock[self.day-1].close[-1] / self.stock[self.day].close[-1])-1 
            if action == -1:
                if self.shares[-1] > 0:
                    # 卖出
                    cash = self.stock[self.day-1].close.values[-1] * self.shares[-1] * (1 - self.transaction_cost)
                    self.balance += cash
                    # 更改份额
                    self.shares[-1] = 0
                    self.reward = np.log(self.balance + self.stock[self.day].close.values[-1] * self.shares[-1] - begin_value)



