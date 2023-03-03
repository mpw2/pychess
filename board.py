
BOARD_SIZE = 8
FILE_NAME = "abcdefgh"

PAWN = 1
KNIGHT = 3
BISHOP = 5
ROOK = 7
QUEEN = 9
KING = 11

WHITE = 1
BLACK = 0

class Board:
    
    def __init__(self):
        self.squares = [[0 for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
        self.moveNum = 0
        self.setUpPieces()

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
            print("Not implemented")

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
                    moves.append(FILE_NAME[j]+str(i+1)+FILE_NAME[j]+str(i+2))
                if i == 1 and not self.hasPiece(i+1,j) \
                     and not self.hasPiece(i+2,j):
                    moves.append(FILE_NAME[j]+str(i+1)+FILE_NAME[j]+str(i+3))
                if self.hasPiece(i+1,j+1) and self.color(i+1,j+1) == BLACK:
                    moves.append(FILE_NAME[j]+str(i+1)+"x"+FILE_NAME[j+1]+str(i+2))
                if self.hasPiece(i+1,j-1) and self.color(i+1,j-1) == BLACK:
                    moves.append(FILE_NAME[j]+str(i+1)+"x"+FILE_NAME[j-1]+str(i+2))
            else: # c == BLACK
                if not self.hasPiece(i-1,j) and i > 0:
                    moves.append(FILE_NAME[j]+str(i+1)+FILE_NAME[j]+str(i))
                if i == BOARD_SIZE-2 and not self.hasPiece(i-1,j) \
                     and not self.hasPiece(i-2,j):
                    moves.append(FILE_NAME[j]+str(i+1)+FILE_NAME[j]+str(i-1))
                if self.hasPiece(i-1,j+1) and self.color(i-1,j+1) == WHITE:
                    moves.append(FILE_NAME[j]+str(i+1)+"x"+FILE_NAME[j+1]+str(i))
                if self.hasPiece(i-1,j-1) and self.color(i-1,j-1) == WHITE:
                    moves.append(FILE_NAME[j]+str(i+1)+"x"+FILE_NAME[j-1]+str(i))
                    
        elif self.isPiece(s,KNIGHT):
            iOffset = [2,2,-2,-2,1,-1,1,-1]
            jOffset = [1,-1,1,-1,2,2,-2,-2]
            for io,jo in zip(iOffset,jOffset):
                if not self.inBounds(i+io,j+jo):
                    continue
                if not self.hasPiece(i+io,j+jo):
                    moves.append("N"+FILE_NAME[j]+str(i+1)+FILE_NAME[j+jo]+str(i+io+1))
                elif self.color(i,j) != self.color(i+io,j+jo):
                    moves.append("N"+FILE_NAME[j]+str(i+1)+"x"+FILE_NAME[j+jo]+str(i+io+1))

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
                        moves.append("B"+FILE_NAME[j]+str(i+1)+FILE_NAME[j+jo*step]+str(i+io*step+1))
                        step += 1
                        continue
                    elif self.color(i,j) == self.color(i+io*step,j+jo*step):
                        step = -1
                        continue
                    elif self.color(i,j) != self.color(i+io*step,j+jo*step):
                        moves.append("B"+FILE_NAME[j]+str(i+1)+"x"+FILE_NAME[j+jo*step]+str(i+io*step+1))
                        step = -1
                        continue

        elif self.isPiece(s,ROOK):
            pass
        elif self.isPiece(s,QUEEN):
            pass
        elif self.isPiece(s,KING):
            pass

        return moves

    def makeMove(self,notation):
        ij12 = [0,0,0,0]
        c = 0
        for s in notation:
            if s in "NBRQKx":
                pass
            elif s in FILE_NAME:
                ij12[c] = FILE_NAME.index(s)
                c += 1
            else:
                ij12[c] = int(s)-1
                c += 1
        if c < 4:
            return False
        value = self.squares[ij12[1]][ij12[0]]
        self.squares[ij12[3]][ij12[2]] = value
        self.squares[ij12[1]][ij12[0]] = 0
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
        fil = FILE_NAME[j]
        rnk = i+1
        return fil+str(rnk)

    def not2idx(self,a1):
        fil = FILE_NAME.index(a1[0])
        rnk = int(a1[1])-1
        return (rnk,fil)

    def __str__(self):
        out = ""
        row1 = ""
        row2 = ""
        row3 = ""
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                s = self.squares[BOARD_SIZE-i-1][j]
                piece_name = "."
                if self.isPiece(s,PAWN):
                    piece_name = FILE_NAME[j]
                elif self.isPiece(s,KNIGHT):
                    piece_name = "N"
                elif self.isPiece(s,BISHOP):
                    piece_name = "B"
                elif self.isPiece(s,ROOK):
                    piece_name = "R"
                elif self.isPiece(s,QUEEN):
                    piece_name = "Q"
                elif self.isPiece(s,KING):
                    piece_name = "K"

                if self.isColor(s,WHITE):
                    row1 += "---"
                    row2 += "-"+piece_name+"-"
                    row3 += "---"
                elif self.isColor(s,BLACK):
                    row1 += piece_name+piece_name+piece_name
                    row2 += piece_name+piece_name+piece_name
                    row3 += piece_name+piece_name+piece_name
                else:
                    row1 += "..."
                    row2 += "..."
                    row3 += "..."
            out += row1 + "\n"
            out += row2 + "\n"
            out += row3 + "\n"
            row1 = ""
            row2 = ""
            row3 = ""
        return out

    def __enter__(self):
        return self

def main():
    B = Board()
    assert( B.idx2not(1,1) == "b2" )
    assert( B.not2idx("a1") == (0,0) )
    print( B.allLegalMoves() )
    print( B.makeMove("e2e4") )
    print( B.legalMoves(0,5) )
    print( B.makeMove("Bf1c4") )
    print( B.legalMoves(3,2) )

if __name__ == "__main__":
    main()
