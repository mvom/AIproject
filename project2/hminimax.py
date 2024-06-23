from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.util import manhattanDistance
import numpy as np


def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """

    return (
        state.getPacmanPosition(),
        state.getFood(),
        state.getGhostPosition(1)
    )


class PacmanAgent(Agent):
    """
    Pacman agent based on an minimax approach.
    """

    def __init__(self):
        super().__init__()
        self.moves = None
        self.max_depth = 4

    def heuristic(self, state):
        """Returns an estimated utility value of the state.
           the value is based off a linear combination of variables
           such as: distance to food, distance from ghost,
           how much food will be eaten
        Arguments:
        ----------
        - `state`: the current game state.
        Returns:
        ----------
        An estimated utility value
        """

        pacman_position = state.getPacmanPosition()
        ghost_position = state.getGhostPosition(1)
        remaining_food = state.getFood().asList()
        current_score = state.getScore()

        if state.isWin():
            return current_score
        if state.isLose():
            return 0
        # Calculate distance from pacman to nearest food
        distance_food = []
        for i in remaining_food:
            distance_food.append(manhattanDistance(pacman_position, i))
        avg_distance_food = np.average(distance_food)

        # Calculate the number of remaining food
        number_food = len(remaining_food)

        # Calculate the distance from ghost
        distance_ghost = manhattanDistance(pacman_position, ghost_position)

        # Specify the weights for each variable in the score
        weight_1 = 1
        weight_2 = 6
        weight_3 = 1

        # Calculate the score as a linear combination of each attribute
        prod1 = weight_1 * avg_distance_food
        prod2 = weight_2 * number_food
        prod3 = weight_3 * (1/distance_ghost)
        score = current_score - prod1 - prod2 + prod3

        # Convert score to a value between 1 and 0,
        # to fit between the utility function
        value = 1/(1 + np.exp(-score/100))

        return value

    def maxFunction(self, state, next_state, depth):
        if state.isWin():
            return self.heuristic(state)
        if state.isLose():
            return self.heuristic(state)
        if depth == self.max_depth:
            return self.heuristic(state)
        # set inital values
        value = 0
        depth = depth + 1
        next_state.add(key(state))
        successors = state.generatePacmanSuccessors()
        for successor in successors:
            next_next_state = next_state.copy()
            aux = self.minFunction(successor[0], next_next_state, depth)
            value = max(value, aux)
        return value

    def minFunction(self, state, next_state, depth):

        if state.isWin():
            return self.heuristic(state)
        if state.isLose():
            return self.heuristic(state)
        if depth == self.max_depth:
            return self.heuristic(state)
        # set inital values
        value = 1
        depth = depth + 1

        next_state.add(key(state))

        successors = state.generateGhostSuccessors(1)
        for successor in successors:
            next_next_state = next_state.copy()
            aux = self.maxFunction(successor[0], next_next_state, depth)
            value = min(value, aux)
        return value

    def get_action(self, state):
        """ Given a pacman game state, returns a legal move
            Arguments:
                state: the current game state
            Returns:
                A legal move as defined in 'game.Directions'.
        """
        # Initialise values
        current_best_value = 0
        current_best_action = Directions.STOP
        depth = 0

        past_states = set()

        past_states.add(key(state))

        successors = state.generatePacmanSuccessors()
        for next_state, next_action in successors:
            value = self.minFunction(next_state, past_states, depth)
            if value > current_best_value:
                current_best_value = value
                current_best_action = next_action

        return current_best_action
