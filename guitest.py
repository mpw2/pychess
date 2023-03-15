
import tkinter as tk
from PIL import ImageTk, Image

import board
from constants import *

class chessGUI:
    def __init__(self):
        self.window = tk.Tk()

        self.window.resizable(False,False)

        self.board_frame = tk.Frame(master=window, width=BOARD_PIXELS, 
                height=BOARD_PIXELS, bg=None)
        self.board_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=False)

        self.move_frame = tk.Frame(master=window, width=MOVE_FRAME_PIXELS, bg=BACKGROUND_COLOR)
        self.move_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=False)

        squares = [[None for i in range(BOARD_SIZE)] for j in range(BOARD_SIZE)]

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                color = LIGHT_SQUARE_COLOR
                if j%2 == i%2:
                    color = DARK_SQUARE_COLOR
                squares[i][j] = tk.Canvas(master=board_frame, width=SQUARE_PIXELS, height=SQUARE_PIXELS, bg=color)
                squares[i][j].place(x=(j*SQUARE_PIXELS),y=((BOARD_SIZE-i-1)*SQUARE_PIXELS))

def renderBoard(board):
    for rr in range(BOARD_SIZE):
        for ff in range(BOARD_SIZE):
            pass


def main():
    
    window = tk.Tk()
    
    window.resizable(False,False)
    
    board_frame = tk.Frame(master=window, width=BOARD_PIXELS, height=BOARD_PIXELS, bg=None)
    board_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=False)
    
    move_frame = tk.Frame(master=window, width=MOVE_FRAME_PIXELS, bg=BACKGROUND_COLOR)
    move_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=False)
    
    squares = [[None for i in range(BOARD_SIZE)] for j in range(BOARD_SIZE)]
    
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            color = LIGHT_SQUARE_COLOR
            if j%2 == i%2:
                color = DARK_SQUARE_COLOR
            squares[i][j] = tk.Canvas(master=board_frame, width=SQUARE_PIXELS, height=SQUARE_PIXELS, bg=color)
            squares[i][j].place(x=(j*SQUARE_PIXELS),y=((BOARD_SIZE-i-1)*SQUARE_PIXELS))

    B = board.Board()

    img = [[None for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if not B.hasPiece(i,j):
                continue
            if B.squares[i][j] == WHITE+PAWN:
                img[i][j] = ImageTk.PhotoImage(Image.open("assets/white_pawn.png"))
            elif B.squares[i][j] == BLACK+PAWN:
                img[i][j] = ImageTk.PhotoImage(Image.open("assets/black_pawn.png"))
            elif B.squares[i][j] == WHITE+KNIGHT:
                img[i][j] = ImageTk.PhotoImage(Image.open("assets/white_knight.png"))
            elif B.squares[i][j] == BLACK+KNIGHT:
                img[i][j] = ImageTk.PhotoImage(Image.open("assets/black_knight.png"))
            elif B.squares[i][j] == WHITE+BISHOP:
                img[i][j] = ImageTk.PhotoImage(Image.open("assets/white_bishop.png"))
            elif B.squares[i][j] == BLACK+BISHOP:
                img[i][j] = ImageTk.PhotoImage(Image.open("assets/black_bishop.png"))
            elif B.squares[i][j] == WHITE+ROOK:
                img[i][j] = ImageTk.PhotoImage(Image.open("assets/white_rook.png"))
            elif B.squares[i][j] == BLACK+ROOK:
                img[i][j] = ImageTk.PhotoImage(Image.open("assets/black_rook.png"))
            elif B.squares[i][j] == WHITE+QUEEN:
                img[i][j] = ImageTk.PhotoImage(Image.open("assets/white_queen.png"))
            elif B.squares[i][j] == BLACK+QUEEN:
                img[i][j] = ImageTk.PhotoImage(Image.open("assets/black_queen.png"))
            elif B.squares[i][j] == WHITE+KING:
                img[i][j] = ImageTk.PhotoImage(Image.open("assets/white_king.png"))
            elif B.squares[i][j] == BLACK+KING:
                img[i][j] = ImageTk.PhotoImage(Image.open("assets/black_king.png"))

            squares[i][j].create_image(SQUARE_PIXELS//2,SQUARE_PIXELS//2,
                    image=img[i][j])
    
    window.mainloop()

if __name__ == "__main__":
    main()
