import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

def main():
    
    result_file = "results/SimpleAgent_v2-3/rewards.pkl"

    with open(result_file, "rb") as f:
        rewards = pickle.load(f)

    rewards = np.array([r for r in rewards])
    episodes = np.arange(1,rewards.size+1,1)

    plt.plot(episodes[-100:], rewards[-100:])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
    

if __name__ == "__main__":
    main()
