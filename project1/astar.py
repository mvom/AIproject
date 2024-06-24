# Importing necessary modules
from pacman_module.game import Agent, Directions
from pacman_module.util import PriorityQueue, manhattanDistance


def key(state):
    """
    Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple representing the current state.
    """
    return (
        state.getPacmanPosition(),
        state.getFood(),
        tuple(state.getCapsules())
    )


def cost_function(state, next_state):
    """
    Given a pacman game state, returns a value based on the
    cost of moving from one state to another.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.
        next_state: the next game state.

    Returns:
        An integer value for the cost.
    """
    current_capsules = state.getCapsules()
    next_state_capsules = next_state.getCapsules()

    # Cost for moving and eating a capsule
    if current_capsules != next_state_capsules:
        return 6
    else:
        return 1  # Cost for moving


def heuristic_function(state):
    """
    Calculates the Manhattan distances to the closest food positions.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        An integer value that denotes the heuristic value for the position.
    """
    current_position = state.getPacmanPosition()
    food_positions = state.getFood()
    distances = [
        manhattanDistance((x, y), current_position)
        for x in range(food_positions.width)
        for y in range(food_positions.height)
        if food_positions[x][y]
    ]

    if not distances:
        return 0
    else:
        return max(distances)


class PacmanAgent(Agent):
    """
    Pacman agent based on an A* approach,
    decreasing by moving to the next position,
    and defining for each state the Euclidean distance to the nearest food.
    """

    def __init__(self):
        super().__init__()
        self.moves = None

    def get_action(self, state):
        """
        Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        if self.moves is None:
            self.moves = self.astar(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def astar(self, state):
        """
        Given a pacman game state, returns a list of legal moves
        to solve the search layout based on the A* algorithm.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A list of legal moves.
        """
        path = []
        fringe = PriorityQueue()
        fringe.push((state, path, 0.), 0.)
        closed = set()

        while True:
            if fringe.isEmpty():
                return []

            _, (current, path, cost) = fringe.pop()

            if current.isWin():
                return path

            current_key = key(current)

            if current_key in closed:
                continue
            closed.add(current_key)

            for successor, action in current.generatePacmanSuccessors():
                successor_cost = cost + cost_function(current, successor)
                fringe.push((successor, path + [action], successor_cost),
                            successor_cost + heuristic_function(successor))

        return path
