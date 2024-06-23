from pacman_module.game import Agent, Directions
from pacman_module.util import Queue


def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """
    return (
        state.getPacmanPosition(),
        state.getFood()
    )


class PacmanAgent(Agent):
    """Pacman agent based on breadth-first search (BFS)."""

    def __init__(self):
        super().__init__()
        self.moves = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A legal move as defined in `game.Directions`.
        """
        if self.moves is None:
            self.moves = self.bfs(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def bfs(self, state):
        """Given a Pacman game state, returns a list of legal moves to solve
        the search layout.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A list of legal moves.
        """
        path = []   # Initialize an empty list to represent the path
        fringe = Queue()
        fringe.push((state, path))
        closed = set()    # Initialize a set to keep track of visited states

        while not fringe.isEmpty():
            current, path = fringe.pop()
            # Get a unique key for the current state
            current_key = key(current)

            if current.isWin():
                return path

            if current_key in closed:
                continue
            # Add the current state to the set of visited states
            closed.add(current_key)

            for successor_state, action in current.generatePacmanSuccessors():
                fringe.push((successor_state, path + [action]))

        return []    # Return an empty list if no solution is found
