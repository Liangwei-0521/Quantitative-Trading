import pandas as pd


class data:

    def __init__(self, link, window_length, t):
        
        self.b = []
        self.t = t 
        self.data = pd.read_csv(link)
        self.window_length = window_length

    def process(self, ):
        df = self.data.loc[:, ['Close', 'Open', 'High', 'Low', 'RSI', 'ROC', 'CCI', 'MACD', 'EXPMA', 'VMACD']].set_index(self.data['Date'])
        self.b.insert(-1, [df[i-self.window_length:i] for i in range(self.window_length, len(self.data))])
        return self.b

    def train_data(self, ):
        self.b = self.process()
        return self.b[0][:self.t] 

    def trade_data(self, ):
        self.b = self.process()
        return self.b[0][self.t:] 


if __name__ == "__main__":
    D = data(link=r'..\Data\000001_SZ.csv', window_length=15, t=2000)
    b = D.train_data()
    print(type(b), len(b), type(b[100]))
    print(b[100])
