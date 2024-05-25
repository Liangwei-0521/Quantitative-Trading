import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class policy(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, device) -> None:
        super(policy, self).__init__()
        # Dueling Q Network
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


class policy_2(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device) -> None:
        super(policy_2, self).__init__()
        # Deep Q Network
        self.linear1 = nn.Linear(state_dim, hidden_dim).to(device)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.linear4 = nn.Linear(hidden_dim, action_dim).to(device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        action = F.relu(self.linear4(F.relu(self.linear3(x))))
        return action


class IL_Agent(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, device, lr, gamma, target_update, epsilon) -> None:
        super(IL_Agent, self).__init__()
        self.action_dim = action_dim
        self.device = device
        self.Loss = []
        self.count = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update  # 目标网络更新频率
        # 实例化Q网络
        self.Q_net = policy(state_dim, hidden_dim, action_dim, device)
        # 实例化目标Q网络
        self.Target_Q_Net = policy(state_dim, hidden_dim, action_dim, device)
        # 优化器
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=lr)
        # 交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss()

    def take_action(self, x):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim) - 1
        else:
            action = self.Q_net(x)
            action = action.argmax().item() - 1
        return action

    def update(self, transition_dict, expert_transition_dict, lambda_1, lambda_2):
        # Imitative learning 损失结合 TD 损失
        # First: TD损失
        states = torch.stack(transition_dict['states'], 0).squeeze(dim=1).to(self.device)
        # print('states:', states.size(), states.dtype)
        actions = torch.from_numpy(np.array(transition_dict['actions']).astype(np.int64)).view(-1, 1).to(
            self.device)  # .unsqueeze(dim=2)
        # print('action:', actions.size(), actions.dtype)
        reward = torch.from_numpy(np.array(transition_dict['rewards'])).view(-1, 1).to(self.device).type(torch.float32)
        # print('Reward: ', reward.size(), reward.dtype)
        next_states = torch.stack(transition_dict['next_states'], 0).squeeze(dim=1).to(self.device)
        # print('next_state:', next_states.size(), next_states.dtype)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float64).view(-1, 1).to(self.device).type(
            torch.float32)
        # print('done:', dones.size(), dones.dtype)

        # Second: IL 损失
        expert_states = torch.tensor(np.array(expert_transition_dict['states'])).to(self.device).squeeze(dim=1).to(
            torch.float)
        expert_actions = torch.tensor(np.array(expert_transition_dict['actions']), dtype=torch.int64).view(-1, 1).to(
            self.device)

        # 方法1：                           
        # log_probs = torch.log(self.Q_net(expert_states).gather(1, expert_actions))
        # bc_loss = torch.mean(-log_probs)  # 最大似然估计
        # print('专家状态：', expert_states, '智能体状态：', states)
        # print('最大克隆 最小克隆：', max(-log_probs), min(-log_probs), bc_loss, torch.pow(max(-log_probs)-min(-log_probs), 2))

        # 方法2：交叉熵损失
        IL_Q = torch.softmax(self.Q_net(expert_states).float(), dim=-1)
        expert_actions = F.one_hot(expert_actions, num_classes=3).squeeze(dim=1).to(self.device).float()
        bc_loss = self.criterion(IL_Q, expert_actions)

        # DQN 智能体优化
        Q_values = self.Q_net(states).gather(1, actions)
        # print("Q_Values:", Q_values.size(), Q_values.dtype)
        q_targets = self.Target_Q_Net(next_states).max(1)[0].view(-1, 1)
        # print('q_targets:', q_targets.size())
        Q_targets = reward + self.gamma * q_targets * (1 - dones)  # TD误差目标
        # print("Q_target:", Q_targets.size(), Q_targets.dtype)
        # a = torch.pow(max(Q_values-Q_targets)-min(Q_values-Q_targets), 2).detach().item()
        # b = torch.pow(max(-log_probs)-min(-log_probs), 2).detach().item()
        # print(a, b) lambda_1 * torch.mean(F.mse_loss(Q_values, Q_targets)) +
        DQN_Loss = lambda_1 * 0.01 * torch.mean(F.mse_loss(Q_values, Q_targets)) + lambda_2 * bc_loss
        # print(torch.mean(F.mse_loss(Q_values, Q_targets))/a, 0.005*bc_loss)
        # print('Loss:', F.mse_loss(Q_values, Q_targets), torch.mean(F.mse_loss(Q_values, Q_targets)), bc_loss)

        self.optimizer.zero_grad()
        DQN_Loss.backward()
        self.Loss.append(DQN_Loss)
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.Target_Q_Net.load_state_dict(self.Q_net.state_dict())
        self.count += 1


class IL_Agent_2(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, device, lr, gamma, target_update, epsilon) -> None:
        super(IL_Agent_2, self).__init__()
        self.action_dim = action_dim
        self.device = device
        self.Loss = []
        self.count = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update  # 目标网络更新频率
        # 实例化Q网络
        self.Q_net = policy_2(state_dim, hidden_dim, action_dim, device)
        # 实例化目标Q网络
        self.Target_Q_Net = policy_2(state_dim, hidden_dim, action_dim, device)
        # 优化器
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=lr)
        # 交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss()

    def take_action(self, x):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim) - 1
        else:
            action = self.Q_net(x)
            action = action.argmax().item() - 1
        return action

    def update(self, transition_dict, expert_transition_dict, lambda_1, lambda_2):
        # Imitative learning 损失结合 TD 损失
        # First: TD损失
        states = torch.stack(transition_dict['states'], 0).squeeze(dim=1).to(self.device)
        # print('states:', states.size(), states.dtype)
        actions = torch.from_numpy(np.array(transition_dict['actions']).astype(np.int64)).view(-1, 1).to(
            self.device)  # .unsqueeze(dim=2)
        # print('action:', actions.size(), actions.dtype)
        reward = torch.from_numpy(np.array(transition_dict['rewards'])).view(-1, 1).to(self.device).type(torch.float32)
        # print('Reward: ', reward.size(), reward.dtype)
        next_states = torch.stack(transition_dict['next_states'], 0).squeeze(dim=1).to(self.device)
        # print('next_state:', next_states.size(), next_states.dtype)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float64).view(-1, 1).to(self.device).type(
            torch.float32)
        # print('done:', dones.size(), dones.dtype)

        # Second: IL 损失
        expert_states = torch.tensor(np.array(expert_transition_dict['states'])).to(self.device).squeeze(dim=1).to(
            torch.float)
        expert_actions = torch.tensor(np.array(expert_transition_dict['actions']), dtype=torch.int64).view(-1, 1).to(
            self.device)

        # 方法1：                           
        # log_probs = torch.log(self.Q_net(expert_states).gather(1, expert_actions))
        # bc_loss = torch.mean(-log_probs)  # 最大似然估计
        # print('专家状态：', expert_states, '智能体状态：', states)

        # 方法2：  
        IL_Q = torch.softmax(self.Q_net(expert_states).float(), dim=-1)
        expert_actions = F.one_hot(expert_actions, num_classes=3).squeeze(dim=1).to(self.device).float()
        bc_loss = self.criterion(IL_Q, expert_actions)

        # DQN 智能体优化
        Q_values = self.Q_net(states).gather(1, actions)
        # print("Q_Values:", Q_values.size(), Q_values.dtype)
        q_targets = self.Target_Q_Net(next_states).max(1)[0].view(-1, 1)
        # print('q_targets:', q_targets.size())
        Q_targets = reward + self.gamma * q_targets * (1 - dones)  # TD误差目标
        # print("Q_target:", Q_targets.size(), Q_targets.dtype)
        a = torch.pow(max(Q_values - Q_targets) - min(Q_values - Q_targets), 2).detach().item()

        DQN_Loss = lambda_1 * torch.mean(F.mse_loss(Q_values, Q_targets)) / a + lambda_2 * 0.005 * bc_loss
        # print('Loss:', torch.mean(F.mse_loss(Q_values, Q_targets)), bc_loss)

        self.optimizer.zero_grad()
        DQN_Loss.backward()
        self.Loss.append(DQN_Loss)
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.Target_Q_Net.load_state_dict(self.Q_net.state_dict())
        self.count += 1


if __name__ == '__main__':
    net = policy(state_dim=10, hidden_dim=5, action_dim=3, device='cuda:0')
    x = torch.rand(size=(1, 10)).cuda()
    print(net(x))
