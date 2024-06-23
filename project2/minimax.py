from pacman_module.game import Agent
from pacman_module.pacman import Directions


def key(state):
    """Returns a key that uniquely identifies a Pacman game state.
    Arguments:
        state: a game state. See API or class `pacman.GameState`.
    Returns:
        A hashable key tuple.
    """
    return (
        state.getPacmanPosition(),
        state.getNumAgents(),
        state.getFood(),
        state.getGhostPosition(1)
    )


class PacmanAgent(Agent):
    """
    Pacman agent based on a minimax approach.
    """

    def __init__(self):
        super().__init__()
        self.moves = None

    def minimax(self, state, depth, is_maximizing):
        """
        Recursive function for the Minimax algorithm.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            depth: the current depth in the Minimax tree.
            is_maximizing: a boolean indicating whether it's
            a maximizing or minimizing level.

        Returns:
            The minimax value of the state.
        """
        if depth == 0 or state.isWin() or state.isLose():
            return self.utility_function(state)

        if is_maximizing:
            value = float("-inf")
            successors = state.generatePacmanSuccessors()

            for next_state, _ in successors:
                value = max(value, self.minimax(next_state, depth - 1, False))

            return value
        else:
            value = float("inf")
            successors = state.generateGhostSuccessors(1)

            for next_state, _ in successors:
                value = min(value, self.minimax(next_state, depth, True))

            return value

    def utility_function(self, state):
        """
        Assessment function to estimate the value of a game state.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The estimated value of the state.
        """
        if state.isWin():
            return 1
        if state.isLose():
            return 0
        return state.getScore()

    def get_action(self, state):
        """
        Given a Pacman game state, returns a legal move
        using the Minimax algorithm.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        # Initialize values
        current_best_value = float("-inf")
        current_best_action = Directions.STOP
        past_states = set()
        past_states.add(key(state))

        successors = state.generatePacmanSuccessors()
        for next_state, next_action in successors:
            value = self.minimax(next_state, depth=2, is_maximizing=False)
            if value > current_best_value:
                current_best_value = value
                current_best_action = next_action

        return current_best_action
