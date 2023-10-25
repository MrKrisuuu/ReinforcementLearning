import gymnasium as gym
import matplotlib.pyplot as plt
import time
import math
from functools import reduce

from model import DQNAgent, QAgent


def train_agent(agent, epochs=1000):
    # env = gym.make("MountainCar-v0")
    env = gym.make("LunarLander-v2")

    if isinstance(agent, QAgent):
        max_space = env.observation_space.high - env.observation_space.low
        max_space_count = [math.ceil(s / agent.r) for s in max_space]
        max_count = reduce(lambda x, y: x * y, max_space_count)
        print(f"Estimated states: {max_count}")

    agent.set_env(env)
    state, info = env.reset()

    rewards = []
    sizes = []
    start_time = time.time()
    for epoch in range(epochs):
        sum_reward = 0
        if isinstance(agent, QAgent):
            sizes.append(agent.get_size())
        while True:
            action = agent.get_action(state, best=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            sum_reward += reward

            agent.train_agent(state, action, reward, next_state, 1 if terminated or truncated else 0)

            state = next_state
            if terminated or truncated:
                state, info = env.reset()
                rewards.append(sum_reward)
                if epoch%1000 == 0:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Epoch {epoch}: {round(sum(rewards[-1000:]) / 1000, 2)}, {round(elapsed_time, 2)}")
                agent.end_game()
                break
    env.close()
    end_time = time.time()

    agent.save("final")
    if isinstance(agent, QAgent):
        sizes.append(agent.get_size())
        with open(f"sizes_{agent.get_name()}.txt", 'w') as file:
            for size in sizes:
                file.write(str(size) + '\n')

    with open(f"rewards_{agent.get_name()}.txt", 'w') as file:
        for reward in rewards:
            file.write(str(round(reward, 2)) + '\n')

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {round(elapsed_time, 2)} seconds for {agent.get_name()}")
    return 1


def test_agent(agent, epochs=10):
    env = gym.make("LunarLander-v2", render_mode="human")

    agent.set_env(env)
    state, info = env.reset()

    for epoch in range(epochs):
        sum_reward = 0
        while True:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            sum_reward += reward

            state = next_state
            if terminated or truncated:
                state, info = env.reset()
                print(f"Epoch {epoch}: {round(sum_reward, 2)}")
                break
    env.close()


def get_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            dat = float(line.strip())
            data.append(dat)
    return data


def get_avg(values, size):
    s = int(size / 2)
    avgs = []
    p = max(0, -s)
    q = min(len(values), s+1)
    current_sum = sum(values[p:q])
    for i in range(0, len(values)):
        new_p = max(0, i-s)
        new_q = min(len(values), i+s+1)
        if new_p != p:
            current_sum -= values[p]
        if new_q != q:
            current_sum += values[new_q-1]
        avgs.append(current_sum / (new_q-new_p+1))
        p = new_p
        q = new_q
    return avgs


def get_decays(d, epochs):
    decays = [1]
    for _ in range(epochs-1):
        decays.append(max(decays[-1]*d, 0.05))
    return decays


def plot_rewards(rewards, length, pass_reward=200, xlabel="Reward", title="Lunar Lander"):
    plt.scatter(list(range(len(rewards))), rewards)
    sols_epochs = []
    sols = []
    for i, reward in enumerate(rewards):
        if reward >= pass_reward:
            sols_epochs.append(i)
            sols.append(reward)
    plt.scatter(sols_epochs, sols, color="red")
    plt.plot([], [])
    plt.plot(list(range(len(rewards))), get_avg(rewards, length))
    plt.xlabel(xlabel)
    plt.ylabel("Epoch")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_diff(differs, labels, ylabel="Reward", title="Lunar Lander"):
    for diff, label in zip(differs, labels):
        plt.plot(list(range(len(diff))), diff, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
