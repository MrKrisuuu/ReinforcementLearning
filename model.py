import torch
import torch.nn as nn
import numpy as np
import pickle
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class DQNAgent(nn.Module):
    def __init__(self, dim_in, dims_inside, dim_out, lr=0.001, y=0.9, d=False, name="agent"):
        super(DQNAgent, self).__init__()

        self.dtype = torch.float

        self.layer_in = nn.Linear(dim_in, dims_inside[0], dtype=self.dtype)
        self.layers_inside = [nn.Linear(dims_inside[i], dims_inside[i+1], dtype=self.dtype).to(device) for i in range(len(dims_inside)-1)]
        self.layer_out = nn.Linear(dims_inside[-1], dim_out, dtype=self.dtype)
        self.act = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters())

        self.lr = lr
        self.y = y
        self.e = 0.05
        if d:
            self.e = 1
        self.d = d
        self.name = name

        self.memory = []

    def forward(self, x):
        x = x
        x = self.act(self.layer_in(x))
        for layer in self.layers_inside:
            x = self.act(layer(x))
        x = self.layer_out(x)
        return x

    def train_agent(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if done:
            state = torch.tensor(np.array([x[0] for x in self.memory])).to(device)
            action = torch.tensor(np.array([x[1] for x in self.memory])).to(device)
            reward = torch.tensor(np.array([x[2] for x in self.memory])).to(device)
            next_state = torch.tensor(np.array([x[3] for x in self.memory])).to(device)
            done = torch.tensor(np.array([x[4] for x in self.memory])).to(device)

            q_value_state = self.forward(state).type(self.dtype)
            q_value_next_state = self.forward(next_state)

            target = reward + (done-1) * self.y * torch.max(q_value_next_state)

            rows = torch.tensor(range(len(self.memory)))
            q_value_state[rows, action] = target.type(self.dtype)

            loss = nn.MSELoss()(self.forward(state), q_value_state)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.memory = []

    def get_action(self, state, best=True):
        if not best and np.random.rand(1) < self.e:
            return np.random.randint(0, self.env.action_space.n)
        else:
            actions = self.forward(torch.tensor(state).to(device).type(self.dtype))
            return torch.argmax(actions).item()

    def end_game(self):
        if self.d and self.e > 0.05:
            self.e *= self.d

    def set_env(self, env):
        self.env = env

    def save(self, name):
        torch.save(self.state_dict(), f"lunar_lander_{self.get_name()}_{name}.pth")

    def load(self, name):
        self.load_state_dict(torch.load(f"lunar_lander_{self.get_name()}_{name}.pth"))

    def get_name(self):
        return f"DQN_{self.name}"


class QAgent():
    def __init__(self, r=0.2, lr=0.1, y=0.9, u=False, d=False, name="agent"):
        self.Q_table = {}
        self.r = r
        self.lr = lr
        self.y = y
        self.u = u
        self.e = 0.05
        if d:
            self.e = 1
        self.d = d
        self.name = name

    def round_state(self, state):
        Q_state = state
        if self.u:
            Q_state = np.power(np.abs(state), 1/self.u)
        Q_state = Q_state / self.r
        Q_state = np.round(Q_state)
        Q_state = np.round(Q_state * self.r, 2)
        if self.u:
            Q_state = np.round(np.power(Q_state, self.u), 2)
            Q_state = np.where(state < 0, -Q_state, Q_state)
        return tuple(Q_state)

    def get_Q_value(self, state):
        Q_state = self.round_state(state)
        if Q_state in self.Q_table:
            return self.Q_table[Q_state]
        else:
            return np.zeros(self.env.action_space.n)

    def train_agent(self, state, action, reward, next_state, done):
        q_value_state = self.get_Q_value(state)
        q_value_next_state = self.get_Q_value(next_state)

        target = (1 - self.lr) * q_value_state[action] + self.lr * (reward + self.y * (done-1) * max(q_value_next_state))
        q_value_state[action] = target
        self.Q_table[self.round_state(state)] = q_value_state

    def get_action(self, state, best=True):
        if not best and np.random.rand(1) < self.e:
            return np.random.randint(0, self.env.action_space.n)
        else:
            actions = self.get_Q_value(state)
            return np.argmax(actions)

    def end_game(self):
        if self.d and self.e > 0.05:
            self.e *= self.d

    def set_env(self, env):
        self.env = env

    def save(self, name):
        with open(f"lunar_lander_{self.get_name()}_{name}.pkl", 'wb') as file:
            pickle.dump(self.Q_table, file)

    def load(self, name):
        with open(f"lunar_lander_{self.get_name()}_{name}.pkl", 'rb') as file:
            self.Q_table = pickle.load(file)

    def get_size(self):
        return len(self.Q_table)

    def get_name(self):
        return f"Q_{self.name}"
