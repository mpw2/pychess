#!/usr/bin/python3

import board
import random
from constants import *
from agent import Agent
from human_agent import HumanAgent
import config

def main():
    print("=== chess.py ===")

    B = board.Board()

    player = HumanAgent(B)

    engine = Agent(B,config.config_simpleagent())

    n = 0
    last_to_move = None
    while(B.game_over == False):
        n += 1
        
        move1 = None
        legal_moves = B.allLegalMoves(B.toMove)
        move1 = engine.choose_move(legal_moves)

        B.makeMove(move1)
        last_to_move = WHITE

        if B.game_over:
            print(f"{n}. {move1}")
            continue

        move2 = None
        legal_moves = B.allLegalMoves(B.toMove)
        while(move2 is None):
            move2 = player.choose_move()
            if not move2 in legal_moves:
                move2 = None
                print("That is an illegal move")

        B.makeMove(move2)
        last_to_move = BLACK

        print(f"{n}. {move1} {move2}")

    if last_to_move == WHITE:
        print("White wins!")
    else:
        print("Black wins!")


def test():
    B = board.Board()
    print(B)

    n = 0
    while(B.game_over == False):
    #for n in range(N):
        n += 1
        movelist = B.allLegalMoves(1)
        move = random.choice(movelist)
        e = B.makeMove(move)
        if not e:
            print("Move failed...")
        if B.game_over == True:
            print(str(n+1)+". "+move)
            continue
        movelist = B.allLegalMoves(0)
        move2 = random.choice(movelist)
        e = B.makeMove(move2)
        if not e:
            print("Move failed...")
        print(str(n+1)+". "+move+" "+move2)


    B2 = board.Board(B)
    movelist = B2.allLegalMoves(WHITE)
    move = random.choice(movelist)
    e = B2.makeMove(move)
    
    print(B)

    print(move)
    print(B2)

    print("Total # Legal Moves: ",len(B.allLegalMoves()))
    print("White # Legal Moves: ",len(B.allLegalMoves(WHITE)))
    print("Black # Legal Moves: ",len(B.allLegalMoves(BLACK)))

    # test en passant
    B3 = board.Board()
    B3.makeMove("e2e4")
    B3.makeMove("e7e6")
    B3.makeMove("e4e5")
    B3.makeMove("d7d5")
    B3.makeMove("e5xd6")
    B3.makeMove("f7f6")
    B3.makeMove("d6d7")
    B3.makeMove("Ke8f7")
    print(B3.legalMoves(*B3.not2idx("d7")))
    B3.makeMove("d7xc8=Q")
    print(B3.legalMoves(*B3.not2idx("c8")))
    print(B3)

if __name__ == "__main__":
    main()
