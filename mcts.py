'''
Jeremy Pham
A12962840
CSE 150
PA3
'''

from __future__ import absolute_import, division, print_function
from math import sqrt, log
from operator import attrgetter
import random
import copy
from randplay import *

#Feel free to add extra classes and functions

'''
State objects are used to keep track of the current board and which player
will make the next move.
'''
class State:
    def __init__(self, grid, player):
        self.grid = grid
        # player is a char, either 'b' or 'w'
        # this is the player that will make the NEXT move
        self.player = player

'''
This class contains the functionality for the Monte Carlo Tree Search that will
be performed each time the AI requests a new move to execute.
'''
class MCTS:
    def __init__(self, grid, player):
        root_state = State(grid, player)
        # a node is created using a grid and next player in a State object
        self.root = MCNode(root_state)

        # initialize the list of possible actions to keep track of what
        # children can be created when expanding
        self.root.possible_options = self.root.get_possible_options(grid)

    def uct_search(self):
        i = 0
        while i < 1000:
            # tree.root is an MCNode with the root_state
            selected_node = self.selection(self.root)
            #print(terminal_node, "was selected")
            winner = self.simulation(selected_node)
            #print("Winner of the simulation was", winner)
            self.backpropagation(selected_node, winner)
            i += 1

        # get the best child, and return the move that was used to get to that child
        print("Children:", len(self.root.children))
        # for c in self.root.children:
        #         print("Move:", c.prev_move, "Wins:", c.wins, "Visits:", c.visits)
        return self.root.get_best_move()

    '''
    selection() is similar to TreePolicy, decides which node will be used for
    simulation, expanding if necessary by calling expansion()
    '''
    def selection(self, node):
        curr = node
        while curr.terminal == False:
            # if a node can be further expanded, use the newly added node
            # when expanding

            # there are still remaining options, so not fully expanded
            if curr.possible_options:
                #print("Node needs to be expanded")
                return self.expansion(curr)
            # else, no more possible options so fully expanded, choose the best child
            else:
                #print("No expansion, use best child")
                curr = self.best_child(curr)
                #print("Selected", curr.prev_move, "as best child")
        # this node will be used for the rollout
        return curr

    '''
    expansion() takes a node and a list of its next available moves and adds
    a child to the node using the first element of the list
    '''
    def expansion(self, node):
        # possible_options will always be nonempty since a check is done in selection()

        player = node.state.player
        next_player = 'b'
        if(player == 'b'):
            next_player = 'w'

        # options[0] is a (r,c) pair
        new_piece = node.possible_options.pop()
        #print("Expanding with", new_piece)
        r = new_piece[0]
        c = new_piece[1]

        # create a copy of the game grid and add the new piece to it
        new_grid = copy.deepcopy(node.state.grid)
        if new_grid[r][c] == '.':
            new_grid[r][c] = player
        else:
            print("Option is invalid, new child cannot be created")
            return None

        # create the new child as a result of expanding the current node
        new_child = MCNode(State(new_grid, next_player))

        # create a Randplay object in order to check if this new piece ended the game
        # this will be a terminal node if the game is over
        curr_game = Randplay(new_grid, next_player)
        curr_game.check_win(r, c)
        if curr_game.game_over:
            new_child.terminal = True
            new_child.winner = curr_game.winner

        # update vars in the parent and child
        node.children.append(new_child)

        new_child.parent = node
        new_child.prev_move = new_piece

        # initialize the list of possible actions to keep track of what
        # children can be created when expanding
        new_child.possible_options = new_child.get_possible_options(new_grid)

        # no moves left -> also the end of the game (terminal node)
        if not new_child.possible_options:
            new_child.terminal = True
            # this will define the case where the node is terminal, but tie game
            new_child.winner = 'tie'

        return new_child

    def best_child(self, node):
        if not node.children:
            print("There are no children, cannot get best_child")
            return None
        return node.max_UCB()

    def simulation(self, node):
        if node.terminal:
            return node.winner
        else:
            # make a copy since the state of the board is changing
            grid_copy = copy.deepcopy(node.state.grid)
            winner = None

            sim = Randplay(grid_copy, node.state.player)
            # rollout performs random moves starting with node.state.player
            # and returns the winner of the simulation as an array with either
            # b win = [0,1] or w win = [1,0]
            simRes = sim.rollout()

            if simRes['b'] == 0 and simRes['w'] == 1:
                winner = 'b'
            if simRes['b'] == 1 and simRes['w'] == 0:
                winner = 'w'
            return winner

    def backpropagation(self, node, result):
        curr = node
        winner = result
        while curr is not None:
            curr.visits += 1

            # this node was terminal but NO winner do not add any wins
            if winner == 'tie':
                pass
            # if the node at the current depth is not the winner, then the node
            # at one depth above is the winner
            elif winner != curr.state.player:
                curr.wins += 1
            curr = curr.parent
        return

# '''
# This class represents the game tree, which is constructed to determine the best
# move at a particular state of the game. A new tree should be constructed
# each time a move is needed.
# '''
# class MCTree:
#     def __init__(self, root_state):
#         self.root = MCNode(root_state)

'''
This node class represents the nodes of the MCTree
'''
class MCNode:
    def __init__(self, curr_state):
        self.state = curr_state
        self.children = []
        self.parent = None
        self.wins = 0
        self.visits = 0
        self.UCB = 0
        self.prev_move = None
        self.possible_options = []
        self.terminal = False
        # only applies if this is a terminal node
        self.winner = None
        self.maxrc = len(curr_state.grid)-1

    '''
    All possible options include everything within a box that encapsulates
    all the current board pieces + 1 row/col on each side of the box.
    '''
    def get_possible_options(self, grid):
        #collect all occupied spots
        current_pcs = []
        for r in range(len(grid)):
            for c in range(len(grid)):
                if not grid[r][c] == '.':
                    current_pcs.append((r,c))
        #At the beginning of the game, curernt_pcs is empty
        if not current_pcs:
            return [(self.maxrc//2, self.maxrc//2)]
        #Reasonable moves should be close to where the current pieces are
        #Think about what these calculations are doing
        #Note: min(list, key=lambda x: x[0]) picks the element with the min value on the first dimension
        min_r = max(0, min(current_pcs, key=lambda x: x[0])[0]-1)
        max_r = min(self.maxrc, max(current_pcs, key=lambda x: x[0])[0]+1)
        min_c = max(0, min(current_pcs, key=lambda x: x[1])[1]-1)
        max_c = min(self.maxrc, max(current_pcs, key=lambda x: x[1])[1]+1)
        #Options of reasonable next step moves
        options = []
        for i in range(min_r, max_r+1):
            for j in range(min_c, max_c+1):
                if not (i, j) in current_pcs:
                    options.append((i,j))
        if len(options) == 0:
            print("No availble options for AI")
            #In the unlikely event that no one wins before board is filled
            #Make white win since black moved first
            # self.game_over = True
            # self.winner = 'w'
        return options

    '''
    Goes through all children of the current node, updates each UCB and returns
    the node with the greatest UCB (Upper Confidence Bound)
    '''
    def max_UCB(self):
        # calculate UCB for each child
        for child in self.children:
            parent = child.parent
            child.UCB = (child.wins/child.visits) + sqrt((2*log(parent.visits))/child.visits)

        # get the child with the greatest UCB value
        return max(self.children, key=attrgetter('UCB'))

    '''
    Does a calculation similar to max_UCB, but only uses the first half
    of the equation (without the portion inside the square root)
    '''
    def get_best_move(self):
        # calculate UCB for each child
        for child in self.children:
            parent = child.parent
            child.UCB = (child.wins/child.visits)

        # get the child with the greatest UCB value
        return (max(self.children, key=attrgetter('UCB'))).prev_move
