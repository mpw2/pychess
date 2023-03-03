import tkinter as tk
from PIL import ImageTk, Image

SQUARE_PIXELS=64
BOARD_SIZE=8
BOARD_PIXELS=BOARD_SIZE*SQUARE_PIXELS

MOVE_FRAME_PIXELS=300

DARK_SQUARE_COLOR="Green"
LIGHT_SQUARE_COLOR="LightGreen"
BACKGROUND_COLOR="DarkSlateGray"

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

img = ImageTk.PhotoImage(Image.open("assets/white_pawn.png"))
for j in range(BOARD_SIZE):
    i = 1
    squares[i][j].create_image(SQUARE_PIXELS//2,SQUARE_PIXELS//2,image=img)

img2 = ImageTk.PhotoImage(Image.open("assets/black_pawn.png"))
for j in range(BOARD_SIZE):
    i = BOARD_SIZE-2
    squares[i][j].create_image(SQUARE_PIXELS//2,SQUARE_PIXELS//2,image=img2)

window.mainloop()
