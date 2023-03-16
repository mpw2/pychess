import agent
import config
import board
from constants import *

def main():

    B = board.Board()
    player = agent.Agent(B,config.config_simpleagent())

    player.train()

if __name__ == "__main__":
    main()
