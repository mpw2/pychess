from constants import *
import board
from agent import Agent
from random_agent import RandomAgent
import config

def main():
    print("=== evaluate.py ===")

    B = board.Board()

    starter = RandomAgent(B)

    random_player = RandomAgent(B)
    #random_player = Agent(B,config.config_simpleagent(train_iter=5))

    engine = Agent(B,config.config_simpleagent())

    n_games = 100
    win_record = [0,0,0]
    max_moves = 100
   
    for t in range(n_games):
        if (t+1) % 10 == 0:
            print(f"Game {t+1} of {n_games}")   
       
        B.reset()
        last_to_move = None

        n = 0
        # make 2 random moves first
        legal_moves = B.allLegalMoves(B.toMove)
        move = starter.choose_move(legal_moves)
        B.makeMove(move)
        legal_moves = B.allLegalMoves(B.toMove)
        move = starter.choose_move(legal_moves)
        B.makeMove(move)
        n = 1
        while(B.game_over == False):
            n += 1

            legal_moves = B.allLegalMoves(B.toMove)
            move1 = engine.choose_move(legal_moves)

            B.makeMove(move1)
            last_to_move = WHITE

            if B.game_over:
                continue

            legal_moves = B.allLegalMoves(B.toMove)
            move2 = random_player.choose_move(legal_moves)

            B.makeMove(move2)
            last_to_move = BLACK

            if n >= max_moves:
                break

        if B.game_over == False:
            win_record[2] += 1
        elif last_to_move == WHITE:
            win_record[0] += 1
        elif last_to_move == BLACK:
            win_record[1] += 1

    print()
    print("Win Record (White == Engine, Black == Random): ")
    print(f"White: {win_record[0]}/{n_games}")
    print(f"Black: {win_record[1]}/{n_games}")
    print(f"Drawn: {win_record[2]}/{n_games}")
    print()

    win_record = [0,0,0]
    
    for t in range(n_games):
        if (t+1) % 10 == 0:
            print(f"Game {t+1} of {n_games}")   
       
        B.reset()
        last_to_move = None

        n = 0
        # make 2 random moves first
        legal_moves = B.allLegalMoves(B.toMove)
        move = starter.choose_move(legal_moves)
        B.makeMove(move)
        legal_moves = B.allLegalMoves(B.toMove)
        move = starter.choose_move(legal_moves)
        B.makeMove(move)
        last_to_move = None
        n = 1
        while(B.game_over == False):
            n += 1

            legal_moves = B.allLegalMoves(B.toMove)
            move1 = random_player.choose_move(legal_moves)

            B.makeMove(move1)
            last_to_move = WHITE

            if B.game_over:
                continue

            legal_moves = B.allLegalMoves(B.toMove)
            move2 = engine.choose_move(legal_moves)

            B.makeMove(move2)
            last_to_move = BLACK

            if n >= max_moves:
                break

        if B.game_over == False:
            win_record[2] += 1
        elif last_to_move == WHITE:
            win_record[0] += 1
        elif last_to_move == BLACK:
            win_record[1] += 1

    print()
    print("Win Record (White == Random, Black == Engine): ")
    print(f"White: {win_record[0]}/{n_games}")
    print(f"Black: {win_record[1]}/{n_games}")
    print(f"Drawn: {win_record[2]}/{n_games}")
    print()
            

if __name__ == "__main__":
    main()
