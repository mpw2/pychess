#!/usr/bin/python3

import board
import random
from constants import *

def main():
    print("=== chess.py ===")
    B = board.Board()
    print(B)

    N = 10
    for n in range(N):
        movelist = B.allLegalMoves(1)
        move = random.choice(movelist)
        e = B.makeMove(move)
        if not e:
            print("Move failed...")
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
