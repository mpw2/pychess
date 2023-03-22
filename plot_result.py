import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

def main():
    
    result_file = "results/SimpleAgent_v3-3/rewards.pkl"
    loss_file = "results/SimpleAgent_v3-3/losses.pkl"

    with open(result_file, "rb") as f:
        rewards = pickle.load(f)

    with open(loss_file, "rb") as f:
        losses = pickle.load(f)

    rewards = np.array([r for r in rewards])
    losses = np.array([ell for ell in losses])
    episodes = np.arange(1,rewards.size+1,1)
    episodes2 = np.arange(1,losses.size+1,1)

    plt.plot(episodes[-100:], rewards[-100:])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

    plt.plot(episodes2, losses)
    plt.xlabel("Episode")
    plt.ylabel("Losses")
    plt.show()
    

if __name__ == "__main__":
    main()
