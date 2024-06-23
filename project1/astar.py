from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.util import PriorityQueue, manhattanDistance


def key(state):
    """
    Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """
    return (
        state.getPacmanPosition(),
        state.getFood(),
        tuple(state.getCapsules())
    )  # Generating a unique key for the state


def cost_function(state, next_state):
    """
    Calculates the transition cost between Pac-Man game states.

    Arguments:
        state, next_state: a game state. See API or class `pacman.GameState`.

    Returns:
        An integer value for the cost.
    """

    return 6 if state.getCapsules() != next_state.getCapsules() else 1


def heuristic_function(state):
    """Heuristic function for Pacman game state."""
    current_position = state.getPacmanPosition()
    food_positions = state.getFood()

    # Calculate the distance to the closest food using Euclidean distance
    closest_food_distance = float('inf')
    for x in range(food_positions.width):
        for y in range(food_positions.height):
            if food_positions[x][y]:
                distance = manhattanDistance((x, y), current_position)
                if distance < closest_food_distance:
                    closest_food_distance = distance

    # Calculate the distance to the closest capsule (if any)
    capsules = state.getCapsules()
    closest_capsule_distance = float('inf')
    for capsule in capsules:
        distance = manhattanDistance(capsule, current_position)
        if distance < closest_capsule_distance:
            closest_capsule_distance = distance

    return (max(closest_food_distance, closest_capsule_distance)
            if closest_food_distance != float('inf') else 0)


class PacmanAgent(Agent):
    """Pacman agent based on A*."""

    def __init__(self):

        super().__init__()
        self.moves = []  # List to store the computed moves

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A legal move as defined in `game.Directions`.
        """

        if not self.moves:
            self.moves = self.astar(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def astar(self, state):
        """Given a pacman game state, returns a list of legal moves
            to solve the search layout. These moves are founded by a bfs
            search algortihm.

        Arguments:
            state: a game state, See API or class pacman.GameState`.

        Returns:
            A list of legal moves.
        """
        path = []
        fringe = PriorityQueue()
        fringe.push((state, path, 0.), 0.)
        closed = set()

        while not fringe.isEmpty():
            # Pop the state with the lowest priority from the fringe
            current, path, cost = fringe.pop()[1]

            if current.isWin():
                return path  # Return the path if the goal state is reached

            current_key = key(current)

            if current_key in closed:
                continue  # Skip the state if it has already been visited

            closed.add(current_key)
            successors = current.generatePacmanSuccessors()
            for successor, action in successors:
                successor_cost = cost + cost_function(current, successor)
                fringe.push((successor, path + [action], successor_cost),
                            successor_cost + heuristic_function(successor))

        return []  # Return an empty path if no path is found
