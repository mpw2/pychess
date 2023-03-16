from constants import *
import numpy as np

class Board:
    
    def __init__(self,position=None):
        self.squares = [[0 for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
        self.setUpPieces(position)
        if not position is None:
            self.moveNum = position.moveNum
            self.toMove = position.toMove
            self.enPassantTarget = position.enPassantTarget
            self.game_over = position.game_over
        else:
            self.moveNum = 0
            self.toMove = WHITE
            self.enPassantTarget = None
            self.game_over = False

    def reset(self):
        self.squares = [[0 for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
        self.moveNum = 0
        self.setUpPieces()
        self.toMove = WHITE
        self.enPassantTarget = None
        self.game_over = False

    def setUpPieces(self,position=None):
        if position is None:
            # set up starting position
            self.squares[0] = \
                    [ROOK,KNIGHT,BISHOP,QUEEN,KING,BISHOP,KNIGHT,ROOK]
            self.squares[1] = [PAWN]*8
            for i in range(BOARD_SIZE):
                self.squares[0][i] += WHITE
                self.squares[1][i] += WHITE
            self.squares[BOARD_SIZE-1] = \
                    [ROOK,KNIGHT,BISHOP,QUEEN,KING,BISHOP,KNIGHT,ROOK]
            self.squares[BOARD_SIZE-2] = [PAWN]*8
            for i in range(BOARD_SIZE):
                self.squares[BOARD_SIZE-1][i] += BLACK
                self.squares[BOARD_SIZE-2][i] += BLACK

        else:
            # set up position from tuple
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    self.squares[i][j] = position.squares[i][j]

    def allLegalMoves(self,turn=None):
        moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if not self.hasPiece(i,j):
                    continue
                if not turn is None and not self.color(i,j) == turn:
                    continue
                moves = moves + self.legalMoves(i,j)
        return moves
                

    def legalMoves(self,i,j):
        moves = []
        if not self.hasPiece(i,j):
            return moves
        s = self.squares[i][j]
        c = self.color(i,j)
        if self.isPiece(s,PAWN):
            if c == WHITE:
                if not self.hasPiece(i+1,j) and i < BOARD_SIZE-1:
                    moves.append(self.idx2not(i,j)+self.idx2not(i+1,j))
                if i == 1 and not self.hasPiece(i+1,j) \
                        and not self.hasPiece(i+2,j):
                    moves.append(self.idx2not(i,j)+self.idx2not(i+2,j))
                if self.hasPiece(i+1,j+1) and self.color(i+1,j+1) == BLACK:
                    moves.append(self.idx2not(i,j)+"x"+self.idx2not(i+1,j+1))
                if self.inBounds(i+1,j+1):
                    if self.enPassantTarget == self.idx2not(i+1,j+1):
                        moves.append(self.idx2not(i,j)+"x"+self.idx2not(i+1,j+1))
                if self.hasPiece(i+1,j-1) and self.color(i+1,j-1) == BLACK:
                    moves.append(self.idx2not(i,j)+"x"+self.idx2not(i+1,j-1))
                if self.inBounds(i+1,j-1):
                    if self.enPassantTarget == self.idx2not(i+1,j-1):
                        moves.append(self.idx2not(i,j)+"x"+self.idx2not(i+1,j-1))
                if len(moves) > 0 and i == BOARD_SIZE-2:
                    promotions = ["=Q","=R","=B","=N"]
                    moves = [m+p for m in moves for p in promotions]

            else: # c == BLACK
                if not self.hasPiece(i-1,j) and i > 0:
                    moves.append(self.idx2not(i,j)+self.idx2not(i-1,j))
                if i == BOARD_SIZE-2 and not self.hasPiece(i-1,j) \
                        and not self.hasPiece(i-2,j):
                    moves.append(self.idx2not(i,j)+self.idx2not(i-2,j))
                if self.hasPiece(i-1,j+1) and self.color(i-1,j+1) == WHITE:
                    moves.append(self.idx2not(i,j)+"x"+self.idx2not(i-1,j+1))
                if self.inBounds(i-1,j+1):
                    if self.enPassantTarget == self.idx2not(i-1,j+1):
                        moves.append(self.idx2not(i,j)+"x"+self.idx2not(i-1,j+1))
                if self.hasPiece(i-1,j-1) and self.color(i-1,j-1) == WHITE:
                    moves.append(self.idx2not(i,j)+"x"+self.idx2not(i-1,j-1))
                if self.inBounds(i-1,j-1):
                    if self.enPassantTarget == self.idx2not(i-1,j-1):
                        moves.append(self.idx2not(i,j)+"x"+self.idx2not(i-1,j-1))
                if len(moves) > 0 and i == 1:
                    promotions = ["=Q","=R","=B","=N"]
                    moves = [m+p for m in moves for p in promotions]
                    
        elif self.isPiece(s,KNIGHT):
            iOffset = [2,2,-2,-2,1,-1,1,-1]
            jOffset = [1,-1,1,-1,2,2,-2,-2]
            for io,jo in zip(iOffset,jOffset):
                if not self.inBounds(i+io,j+jo):
                    continue
                if not self.hasPiece(i+io,j+jo):
                    moves.append("N"+self.idx2not(i,j)+self.idx2not(i+io,j+jo))
                elif self.color(i,j) != self.color(i+io,j+jo):
                    moves.append("N"+self.idx2not(i,j)+"x"+self.idx2not(i+io,j+jo))

        elif self.isPiece(s,BISHOP):
            iOffset = [1,1,-1,-1]
            jOffset = [1,-1,1,-1]
            for io,jo in zip(iOffset,jOffset):
                step = 1
                while(step > 0):
                    if not self.inBounds(i+io*step,j+jo*step):
                        step = -1
                        continue
                    elif not self.hasPiece(i+io*step,j+jo*step):
                        moves.append("B"+self.idx2not(i,j)+self.idx2not(i+io*step,j+jo*step))
                        step += 1
                        continue
                    elif self.color(i,j) == self.color(i+io*step,j+jo*step):
                        step = -1
                        continue
                    elif self.color(i,j) != self.color(i+io*step,j+jo*step):
                        moves.append("B"+self.idx2not(i,j)+"x"+self.idx2not(i+io*step,j+jo*step))
                        step = -1
                        continue

        elif self.isPiece(s,ROOK):
            iOffset = [1,-1,0,0]
            jOffset = [0,0,1,-1]
            for io,jo in zip(iOffset,jOffset):
                step = 1
                while(step > 0):
                    if not self.inBounds(i+io*step,j+jo*step):
                        step = -1
                        continue
                    elif not self.hasPiece(i+io*step,j+jo*step):
                        moves.append("R"+self.idx2not(i,j)+self.idx2not(i+io*step,j+jo*step))
                        step += 1
                        continue
                    elif self.color(i,j) == self.color(i+io*step,j+jo*step):
                        step = -1
                        continue
                    elif self.color(i,j) != self.color(i+io*step,j+jo*step):
                        moves.append("R"+self.idx2not(i,j)+"x"+self.idx2not(i+io*step,j+jo*step))
                        step = -1
                        continue

        elif self.isPiece(s,QUEEN):
            iOffset = [1,-1,0,0,1,1,-1,-1]
            jOffset = [0,0,1,-1,1,-1,1,-1]
            for io,jo in zip(iOffset,jOffset):
                step = 1
                while(step > 0):
                    if not self.inBounds(i+io*step,j+jo*step):
                        step = -1
                        continue
                    elif not self.hasPiece(i+io*step,j+jo*step):
                        moves.append("Q"+self.idx2not(i,j)+self.idx2not(i+io*step,j+jo*step))
                        step += 1
                        continue
                    elif self.color(i,j) == self.color(i+io*step,j+jo*step):
                        step = -1
                        continue
                    elif self.color(i,j) != self.color(i+io*step,j+jo*step):
                        moves.append("Q"+self.idx2not(i,j)+"x"+self.idx2not(i+io*step,j+jo*step))
                        step = -1
                        continue

        elif self.isPiece(s,KING):
            iOffset = [1,-1,0,0,1,1,-1,-1]
            jOffset = [0,0,1,-1,1,-1,1,-1]
            for io,jo in zip(iOffset,jOffset):
                if not self.inBounds(i+io,j+jo):
                    continue
                elif not self.hasPiece(i+io,j+jo):
                    moves.append("K"+self.idx2not(i,j)+self.idx2not(i+io,j+jo))
                    continue
                elif self.color(i,j) == self.color(i+io,j+jo):
                    continue
                elif self.color(i,j) != self.color(i+io,j+jo):
                    moves.append("K"+self.idx2not(i,j)+"x"+self.idx2not(i+io,j+jo))
                    continue


        return moves

    def makeMove(self,notation):
        ij12 = [0,0,0,0]
        c = 0
        isPawn = True
        isCapture = False
        isPromotion = False
        for s in notation:
            if isPromotion:
                promoteTo = s
                continue
            if s in "NBRQK":
                isPawn = False
            elif s in "x":
                isCapture = True
            elif s in "=":
                isPromotion = True
            elif s in FILE_NAME:
                ij12[c] = FILE_NAME.index(s)
                c += 1
            else:
                ij12[c] = int(s)-1
                c += 1
        if c < 4:
            return False

        if isCapture:
            captured_piece = self.squares[ij12[3]][ij12[2]]
            if self.isPiece(captured_piece,KING):
                self.game_over = True

        if not self.hasPiece(ij12[1],ij12[0]):
            return False
        if self.color(ij12[1],ij12[0]) != self.toMove:
            return False

        if self.idx2not(ij12[3],ij12[2]) == self.enPassantTarget:
            if self.toMove == WHITE:
                self.squares[ij12[3]-1][ij12[2]] = 0
            else:
                self.squares[ij12[3]+1][ij12[2]] = 0

        self.enPassantTarget = None

        if isPawn and abs(ij12[3]-ij12[1]) > 1:
            if self.toMove == WHITE:
                self.enPassantTarget = self.idx2not(ij12[3]-1,ij12[2])
            else:
                self.enPassantTarget = self.idx2not(ij12[3]+1,ij12[2])

        value = self.squares[ij12[1]][ij12[0]]
        if isPromotion:
            c = WHITE
            if self.isColor(value,BLACK):
                c = BLACK
            if promoteTo == "Q":
                value = c + QUEEN
            elif promoteTo == "R":
                value = c + ROOK
            elif promoteTo == "B":
                value = c + BISHOP
            elif promoteTo == "R":
                value = c + ROOK

        self.squares[ij12[3]][ij12[2]] = value
        self.squares[ij12[1]][ij12[0]] = 0

        if self.toMove == WHITE:
            self.toMove = BLACK
        else:
            self.toMove = WHITE

        self.moveNum += 1

        return True

    def hasPiece(self,i,j):
        if i < 0 or i >= BOARD_SIZE:
            return False
        if j < 0 or j >= BOARD_SIZE:
            return False
        return ( self.squares[i][j] > 0 )

    def color(self,i,j):
        if not self.hasPiece(i,j):
            return None
        if (self.squares[i][j]+1) % 2 == WHITE:
            return WHITE
        else:
            return BLACK
    
    def isPiece(self,a,b):
        return (a+1)//2 == (b+1)//2

    def isColor(self,a,b):
        if a == 0:
            return False
        return ((a+1)%2) == b

    def inBounds(self,i,j):
        r = True
        if (i < 0 or j < 0):
            r = False
        if (i >= BOARD_SIZE or j >= BOARD_SIZE):
            r = False
        return r

    def idx2not(self,i,j):
        if not self.inBounds(i,j):
            return "Out of bounds"
        fil = FILE_NAME[j]
        rnk = i+1
        return fil+str(rnk)

    def not2idx(self,a1):
        fil = FILE_NAME.index(a1[0])
        rnk = int(a1[1])-1
        return (rnk,fil)

    def flatten(self):
        state = np.zeros(BOARD_SIZE*BOARD_SIZE)
        cnt = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                state[cnt] = self.squares[i][j]
                cnt += 1
        return state

    def __str__(self):
        out = ""
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                s = self.squares[BOARD_SIZE-i-1][j]
                if s == WHITE + PAWN:
                    out += "P"
                elif s == BLACK + PAWN:
                    out += "p"
                elif s == WHITE + KNIGHT:
                    out += "N"
                elif s == BLACK + KNIGHT:
                    out += "n"
                elif s == WHITE + BISHOP:
                    out += "B"
                elif s == BLACK + BISHOP:
                    out += "b"
                elif s == WHITE + ROOK:
                    out += "R"
                elif s == BLACK + ROOK:
                    out += "r"
                elif s == WHITE + QUEEN:
                    out += "Q"
                elif s == BLACK + QUEEN:
                    out += "q"
                elif s == WHITE + KING:
                    out += "K"
                elif s == BLACK + KING:
                    out += "k"
                else:
                    out += "."
                out += " "
            out += "\n"
        return out

    
    # def __str__(self):
    #     out = ""
    #     row1 = ""
    #     row2 = ""
    #     row3 = ""
    #     for i in range(BOARD_SIZE):
    #         for j in range(BOARD_SIZE):
    #             s = self.squares[BOARD_SIZE-i-1][j]
    #             piece_name = "."
    #             if self.isPiece(s,PAWN):
    #                 piece_name = FILE_NAME[j]
    #             elif self.isPiece(s,KNIGHT):
    #                 piece_name = "N"
    #             elif self.isPiece(s,BISHOP):
    #                 piece_name = "B"
    #             elif self.isPiece(s,ROOK):
    #                 piece_name = "R"
    #             elif self.isPiece(s,QUEEN):
    #                 piece_name = "Q"
    #             elif self.isPiece(s,KING):
    #                 piece_name = "K"

    #             if self.isColor(s,WHITE):
    #                 row1 += "---"
    #                 row2 += "-"+piece_name+"-"
    #                 row3 += "---"
    #             elif self.isColor(s,BLACK):
    #                 row1 += piece_name+piece_name+piece_name
    #                 row2 += piece_name+piece_name+piece_name
    #                 row3 += piece_name+piece_name+piece_name
    #             else:
    #                 row1 += "..."
    #                 row2 += "..."
    #                 row3 += "..."
    #         out += row1 + "\n"
    #         out += row2 + "\n"
    #         out += row3 + "\n"
    #         row1 = ""
    #         row2 = ""
    #         row3 = ""
    #     return out

    def __enter__(self):
        return self

def main():
    B = Board()
    assert( B.idx2not(1,1) == "b2" )
    assert( B.not2idx("a1") == (0,0) )
    print( B.allLegalMoves() )
    print( B.makeMove("e2e4") )
    print( B.allLegalMoves() )
    print( B.legalMoves(*B.not2idx("f1")) )
    print( B.makeMove("Bf1c4") )

if __name__ == "__main__":
    main()
