# Quantitative-Trading
Single stock trading strategies based on DQN, DDQN, Behavioural cloning and BCDDQN. 

## Technology Stack

- Pytorch
- Request

## Result

![项目截图](https://github.com/1998-Chen/Quantitative-Trading/blob/main/Result/000001_SZ.svg)

## Code examples
#### Trading Logic
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

## Copyright information

If you like this project, please cite "Deep reinforcement learning stock trading strategy considering behavioral cloning" from the Journal of Systems Management.

## Corresponding author

zhangy@gdut.edu.cn

## Support

This research was supported by the Guangdong Basic and Applied Basic Research Foundation (No. 2023A1515012840)
