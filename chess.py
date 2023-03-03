#!/usr/bin/python3

import board
import random

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

    print(B)
    print("Total # Legal Moves: ",len(B.allLegalMoves()))
    print("White # Legal Moves: ",len(B.allLegalMoves(1)))
    print("Black # Legal Moves: ",len(B.allLegalMoves(0)))

if __name__ == "__main__":
    main()
