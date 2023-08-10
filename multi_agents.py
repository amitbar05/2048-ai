import argparse # debug

import numpy as np
import abc
import util
import math  # we added it
from game import Agent, Action
from game_state import GameState  # we added it


def calc_avg(board, max_tile):
    """
    return total_tiles_worth / number_tiles
    """
    sum = 0
    num_tiles = 0
    for row in board:
        for cell in row:
            if cell != 0:
                sum += cell
                num_tiles += 1
    avg = sum / num_tiles
    board_size = len(board[0]) * len(board)
    return avg + (max_tile / board_size)


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """
        # Useful information you can extract from a GameState (game_state.py)
        # successor_game_state = current_game_state.generate_successor(action=action)
        # board = successor_game_state.board
        # max_tile = successor_game_state.max_tile
        # score = successor_game_state.score

        successor_game_state = current_game_state.generate_successor(action=action)
        avg = calc_avg(successor_game_state.board, successor_game_state.max_tile)
        if action == Action.UP:
            return 4 * avg
        if action == Action.RIGHT:
            return 4 * avg
        if action == Action.DOWN:
            next_legal_actions = successor_game_state.get_legal_actions(0)
            if Action.RIGHT in next_legal_actions:
                next_next = successor_game_state.generate_successor(action=Action.RIGHT)
                return 4 * calc_avg(next_next.board, next_next.max_tile)
        if action == Action.LEFT:
            next_legal_actions = successor_game_state.get_legal_actions(0)
            if Action.UP in next_legal_actions:
                next_next = successor_game_state.generate_successor(action=Action.UP)
                return 4 * calc_avg(next_next.board, next_next.max_tile)
        return avg


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth
        self.our_agent = 0
        self.opponent_agent = 1

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):

    def helper(self, state: GameState, current_depth: int, agent_index: int):
        """
        this is a recursive function that help get_action by returning the estimated score for a state
        """
        legal_actions = state.get_legal_actions(agent_index)
        if not legal_actions:
            return self.evaluation_function(state)
        if agent_index == self.opponent_agent:
            if current_depth == self.depth:
                leaves = [state.generate_successor(agent_index, action) for action in legal_actions]
                min_leaf = min(leaves, key=self.evaluation_function)
                return self.evaluation_function(min_leaf)
            else:
                children = [state.generate_successor(agent_index, action) for action in legal_actions]
                return min([self.helper(child, current_depth + 1, self.our_agent) for child in children])
        else:  # agent_index == self.our_agent
            children = [state.generate_successor(agent_index, action) for action in legal_actions]
            return max([self.helper(child, current_depth, self.opponent_agent) for child in children])

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        legal_action = game_state.get_legal_actions(self.our_agent)
        if not legal_action:
            return Action.STOP
        max_value = -1
        max_action = None
        for action in legal_action:
            current_value = self.helper(game_state.generate_successor(self.our_agent, action), 1, self.opponent_agent)
            if max_value < current_value:
                max_action = action
                max_value = current_value
        return max_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def helper(self, state: GameState, current_depth: int, agent_index: int, alpha, beta):
        """
        this is a recursive function that help get_action by returning the estimated score for a state
        """
        if current_depth > self.depth:
            return self.evaluation_function(state)
        legal_actions = state.get_legal_actions(agent_index)
        if not legal_actions:
            return self.evaluation_function(state)
        if agent_index == self.opponent_agent:
            children = [state.generate_successor(agent_index, action) for action in legal_actions]
            for child in children:
                value = self.helper(child, current_depth + 1, self.our_agent, alpha, beta)
                beta = min(beta, value)
                if beta <= alpha:
                    break  # alpha cutoff
            return beta
        else:
            # agent_index == self.our_agent
            children = [state.generate_successor(agent_index, action) for action in legal_actions]
            for child in children:
                value = self.helper(child, current_depth, self.opponent_agent, alpha, beta)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break  # beta cutoff
            return alpha

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        legal_action = game_state.get_legal_actions(self.our_agent)
        if not legal_action:
            return Action.STOP
        max_value = -1
        max_action = None
        for action in legal_action:
            successor = game_state.generate_successor(self.our_agent, action)
            current_value = self.helper(successor, 1, self.opponent_agent, alpha=-1, beta=math.inf)
            if max_value < current_value:
                max_action = action
                max_value = current_value
        return max_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def helper(self, state: GameState, current_depth: int, agent_index: int):
        """
        this is a recursive function that help get_action by returning the estimated score for a state
        """
        if current_depth > self.depth:
            return self.evaluation_function(state)
        legal_actions = state.get_legal_actions(agent_index)
        if not legal_actions:
            return self.evaluation_function(state)
        if agent_index == self.opponent_agent:
            # we choose random move for the successor
            children = [state.generate_successor(agent_index, action) for action in legal_actions]
            return sum([self.helper(child, current_depth + 1, self.our_agent) for child in children]) / len(children)
        else:  # agent_index == self.our_agent
            children = [state.generate_successor(agent_index, action) for action in legal_actions]
            return max([self.helper(child, current_depth, self.opponent_agent) for child in children])

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        legal_action = game_state.get_legal_actions(self.our_agent)
        if not legal_action:
            return Action.STOP
        max_value = -1
        max_action = None
        for action in legal_action:
            current_value = self.helper(game_state.generate_successor(self.our_agent, action), 1, self.opponent_agent)
            if max_value < current_value:
                max_action = action
                max_value = current_value
        return max_action


def get_value(board, point):
    """
    this function return the value from the cell represented in point if exist, -2 otherwise
    """
    if 0 <= point[0] < len(board) and 0 <= point[1] < len(board[point[0]]):
        return board[point[0]][point[1]]
    return -2


def get_equal_neighbors(board, point):
    """
    this function check if any of the neighbors cell as the same value as the current cell.
    its return the number of equal neighbors time the value in the current cell.
    """
    value = board[point[0]][point[1]]
    count = 0
    neighbors = np.array([[1, 0], [0, 1]])
    for neighbor in neighbors:
        if value == get_value(board, point - neighbor):
            count += 1
        if value == get_value(board, point + neighbor):
            count += 1
    return count * value


def evaluate_neighbors_helper(board, row_index, iterator):
    """
    this function sum of the results of get_equal_neighbors to estimate a score for a board based on the neighbors cell
    with identical values.
    """
    sum = 0
    for col_index in iterator:
        sum += get_equal_neighbors(board, np.array([row_index, col_index]))
    return sum


def evaluate_neighbors(current_game_state):
    """
    return score based on how many neighbors we have.
    """
    sum = 0
    board = current_game_state.board

    for row_index in range(0, len(board), 2):
        # we go through all even indexed rows
        sum += evaluate_neighbors_helper(board, row_index, range(0, len(board[row_index]), 2))

    for row_index in range(1, len(board), 2):
        # we go through all odd indexed rows
        sum += evaluate_neighbors_helper(board, row_index, range(1, len(board[row_index]), 2))
    return sum


def get_log_value(board, point):
    """
    return the log 2 of the value in cell at point in board.
    """
    value = board[point[0]][point[1]]
    if value < 2:
        return value
    return math.log2(value)


def heuristic_close_to_corner(board):
    """
    estimate a score for a board based on the numbers in each cell and there proximity to cell (0,0).
    """
    sum = 0
    for row in range(0, len(board)):
        for col in range(0, len(board[row])):
            manhattan_distance = (len(board) - row) + (len(board[row]) - col)
            close_factor = math.pow(3, manhattan_distance) + manhattan_distance
            sum += close_factor * board[row, col]
    # print("the important corner: ", board[0][0])
    # important_corner = board[0][0]
    # log_important_corner = math.log10(important_corner)
    return math.log2(sum)

def monotonicity(current_game_state):
    """
    This heuristic gives a punishing score every time the monotonicity, of every row and column, is violted
    """
    board = current_game_state.board
    bad_score = 0
    for row in board:
        for i in range(1, len(row)):
            if row[i-1] < row[i]:
                bad_score += math.log2(row[i] - row[i-1])

    for col in range(0, len(board[0])):
        for row in range(1, len(board)):
            if board[col, row-1] < board[col, row]:
                bad_score += math.log2(board[col, row] - board[col, row-1])
    penalty = -1 * bad_score
    return penalty

def better_evaluation_function(current_game_state: GameState):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: we have writen multiple estimation functions to calculate the score of board.
    then we have tried to find the weight for them we will get the optimal score.
    we have multiply each one of the scores we got for a board we a weight measuring how important it is.
    the we started to modify the weight to get more optimal solution.
    """

    w1 = 20
    w2 = 20
    w3 = 14
    w4 = 9
    w5 = 17
    w6 = 2

    score_per_tile = current_game_state.score / current_game_state.board.size

    log_max_tile = math.log2(current_game_state.max_tile)
    score_eval = score_per_tile / log_max_tile
    num_empty_tiles = math.sqrt(len(current_game_state.get_empty_tiles()[0]) * log_max_tile)
    neighbors_score = math.sqrt(evaluate_neighbors(current_game_state) / 2)
    corner_value = heuristic_close_to_corner(current_game_state.board)
    monotonicity_penalty = monotonicity(current_game_state)
    total = w1 * score_eval + w2 * log_max_tile + w3 * num_empty_tiles + w4 * neighbors_score + w5 * corner_value + w6 * monotonicity_penalty
    return total

# Abbreviation
better = better_evaluation_function
