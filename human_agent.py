
class HumanAgent:
    def __init__(self,env):

        self.env = env
       
    def choose_move(self):

        print(self.env)
        move = input("Enter a move: ")
        return move
