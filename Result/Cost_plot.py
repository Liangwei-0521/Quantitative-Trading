import matplotlib.pyplot as plt
from pylab import mpl


# 交易成本---换手率折线图

def cost_plot(DQN, DDQN, BC, ILDDQN, Name):
    mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体：解决plot不能显示中文问题
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams['xtick.direction'] = 'in'  # 刻度在里面
    plt.rcParams['ytick.direction'] = 'in'
    x = [0.0, 0.0015, 0.003, 0.0045, 0.006]

    plt.figure(figsize=(8, 5.5))
    # plt.title(Name, fontdict={'family': 'SimSun', 'size': 10})
    plt.grid()
    plt.xticks(fontsize=20)
    plt.xlabel("交易费用率", fontdict={'family': 'SimSun', 'size': 15})  # 横坐标名字
    plt.yticks(fontsize=20)
    plt.ylabel("年化收益率", fontdict={'family': 'SimSun', 'size': 15})  # 纵坐标名字

    plt.plot(x, BC, linewidth=2.5, label="BC", c='blue')  # s-:方形
    plt.plot(x, DQN, linewidth=2.5, label="DQN", c='green')  # o-:圆形
    plt.plot(x, DDQN, linewidth=2.5, label="DDQN",  c='purple')  # s-:方形
    plt.plot(x, ILDDQN, linewidth=2.5, label="BCDDQN", c='r')  # o-:圆形
    plt.legend()  # 图例
    plt.savefig(r'E:\Code_Project\Agent with imitation\Result\Cost_pic\\' + Name + '.svg', format='svg', dpi=150)
    plt.show()


if __name__ == '__main__':
    # 平安银行
    DQN = [-0.011, -0.086, -0.131, -0.178, -0.23]
    DDQN = [0.113, 0.061, 0.035, -0.099, -0.171]
    BC = [0.209, 0.127, 0.053, -0.015, -0.078]
    ILDDQN = [0.628, 0.431, 0.382, 0.245, 0.182]
    cost_plot(DQN, DDQN, BC, ILDDQN, Name='平安银行')

    # 万科A
    DQN = [-0.291, -0.316, -0.326, -0.373, -0.393]
    DDQN = [-0.128, -0.234, -0.319, -0.397, -0.448]
    BC = [-0.207, -0.251, -0.291, -0.328, -0.361]
    ILDDQN = [0.172, 0.093, 0.064, 0.025, -0.038]
    cost_plot(DQN, DDQN, BC, ILDDQN, Name='万科A')

    # 格力电器
    DQN = [-0.271, -0.301, -0.344, -0.358, -0.375]
    DDQN = [0.045, -0.099, -0.229, -0.313, -0.389]
    BC = [0.049, -0.003, -0.052, -0.098, -0.141]
    ILDDQN = [0.151, 0.108, 0.093, 0.043, 0.019]
    cost_plot(DQN, DDQN, BC, ILDDQN, Name='格力电器')

    # 紫光股份
    DQN = [-0.171, -0.226, -0.234, -0.236, -0.268]
    DDQN = [0.133, -0.0003, -0.116, -0.176, -0.256]
    BC = [0.034, 0.013, -0.008, -0.029, -0.048]
    ILDDQN = [0.225, 0.218, 0.153, 0.117, 0.057]
    cost_plot(DQN, DDQN, BC, ILDDQN, Name='紫光股份')
