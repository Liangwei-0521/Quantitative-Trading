import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device) -> None:
        super(Q_Net, self).__init__()
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


class DDQN_Agent(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device, gamma, epsilon, target_update) -> None:
        super(DDQN_Agent, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        # 执行Q网络
        self.Q_Net = Q_Net(state_dim, hidden_dim, action_dim, device)
        # 目标Q网络
        self.Target_Q_Net = Q_Net(state_dim, hidden_dim, action_dim, device)
        self.Q_optimizer = torch.optim.Adam(self.Q_Net.parameters(), lr=lr)
        self.Loss = []
        self.count = 0
        self.target_update = target_update  # 目标网络更新频率

    def take_action(self, x):
        "The policy of agent"
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim) - 1
        else:
            action = self.Q_Net(x)
            action = action.argmax().item() - 1
        return action

    def update(self, transition_dict):
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
        # DQN 智能体优化
        Q_values = self.Q_Net(states).gather(1, actions)
        # print("Q_Values:", Q_values.size(), Q_values.dtype)
        q_targets = self.Target_Q_Net(next_states).max(1)[0].view(-1, 1)
        # print('q_targets:', q_targets.size())
        Q_targets = reward + self.gamma * q_targets * (1 - dones)  # TD误差目标
        # print("Q_target:", Q_targets.size(), Q_targets.dtype)
        DQN_Loss = torch.mean(F.mse_loss(Q_values, Q_targets))
        self.Q_optimizer.zero_grad()
        DQN_Loss.backward()
        self.Loss.append(DQN_Loss)
        self.Q_optimizer.step()

        if self.count % self.target_update == 0:
            self.Target_Q_Net.load_state_dict(self.Q_Net.state_dict())
        self.count += 1
