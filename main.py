import numpy as np
import gym
import random
from tqdm import tqdm
import os
from time import sleep


def train(Q, episodes, epsilon, lr, gamma):
    for i_episode in tqdm(range(episodes)):
        state = env.reset()
        reward, penalties, epochs = 0, 0, 0
        done = False
        while not done:
            # env.render()
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            new_state, reward, done, info = env.step(action)
            Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            state = new_state
            epochs += 1
        if reward == 1:
            epsilon = epsilon * 0.95
    env.close()
    return Q


def evaluate(Q, show):
    done = False
    state = env.reset()
    all_rewards = []
    frames = []
    timestamps = 0
    while not done:
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        state = new_state
        all_rewards.append(reward)

        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward,
            'total_reward': sum(all_rewards)
        }
        )
        timestamps += 1
    if show:
        print_frames(frames)

    return timestamps, reward


def print_frames(frames):
    for i, frame in enumerate(frames):
        os.system('clear')
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        print(f"Total Reward: {frame['total_reward']}")
        sleep(.5)


if __name__ == "__main__":
    environemt = 'Taxi-v3'
    env = gym.make(environemt)

    all_epochs = []
    all_rewards = []

    retrain = False
    second_retrain = False
    show = True

    episodes = 100000
    epsilon = 1
    lr = 0.01
    gamma = 0.95

    if retrain:
        Q = train(np.zeros([env.observation_space.n, env.action_space.n]), episodes, epsilon, lr, gamma)
        np.savetxt(f"{environemt}_Q_table_episodes{episodes}.csv", Q, delimiter=',')

    Q = np.genfromtxt(f"{environemt}_Q_table_episodes{episodes}.csv", delimiter=',')

    if second_retrain:
        Q = train(Q, 500000, 0.2, lr, gamma)
        np.savetxt(f"{environemt}_Q_table_episodes{1000000}.csv", Q, delimiter=',')

    np.genfromtxt(f"{environemt}_Q_table_episodes{100000}.csv", delimiter=',')
    all_timestamps = []
    for i in range(1000):
        timestamps, rewards = evaluate(Q, show)
        all_timestamps.append(timestamps)
        all_rewards.append(rewards)

    print(np.mean(all_timestamps))
    print(sum(all_rewards))

