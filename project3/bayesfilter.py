import numpy as np
from pacman_module.game import Agent, Directions, manhattanDistance
from scipy.stats import binom
from pacman_module.util import Queue


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls, position):
        """
        Builds the transition matrix T_t = P(X_t | X_{t-1})
        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (i, j) | X_{t-1} = (k, l)) for
            the ghost to move from (k, l) to (i, j).
        """
        width, height = walls.width, walls.height

        # Initialize the transition model
        trans_model = np.zeros((width, height, width, height))

        # Factor based on ghost's state
        ghost_state_factors = {"fearless": 1, "afraid": 2, "terrified": 8}
        f = ghost_state_factors.get(self.ghost, 0)

        for w in range(width):
            for h in range(height):
                if walls[w][h]:
                    continue

                # Adjacent cells of (w, h)
                next_steps = [(w-1, h), (w+1, h), (w, h-1), (w, h+1)]
                dist = manhattanDistance((w, h), position)

                for i, j in next_steps:
                    if 0 <= i < width and 0 <= j < height and not walls[i][j]:
                        next_dist = manhattanDistance((i, j), position)
                        if next_dist > dist:
                            trans_model[i, j, w, h] = f
                        else:
                            trans_model[i, j, w, h] = 1

                # Normalize the matrix for the current cell (w, h)
                norm = np.sum(trans_model[:, :, w, h])
                if norm:
                    trans_model[:, :, w, h] /= norm

        return trans_model

    def observation_matrix(self, walls, evidence, position):
        """
        Builds the observation matrix O_t = P(e_t | X_t)
        given a noisy ghost distance evidence
        e_t and the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """
        width, height = walls.width, walls.height
        binomial = binom(4, 0.5)
        sensor_model = np.zeros((width, height))

        pacman_x, pacman_y = position

        for w in range(width):
            for h in range(height):
                if not walls[w][h]:  # Skip walls
                    # Manhattan distance
                    dist = abs(w - pacman_x) + abs(h - pacman_y)
                    noise = evidence - dist
                    sensor_model[w, h] = binomial.pmf(2 + noise)

        return sensor_model

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """
        width = walls.width
        height = walls.height

        T = self.transition_matrix(walls, position)
        Obs = self.observation_matrix(walls, evidence, position)

        new_belief = np.zeros((width, height))
        s = np.zeros((width, height))

        for w in range(width):
            for h in range(height):
                s[w][h] = np.sum(T[w, h, :, :] * belief)
                new_belief[w, h] = s[w, h] * Obs[w, h]

        norm = np.sum(new_belief)
        if norm != 0:
            new_belief /= norm
        return new_belief

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """
        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls, beliefs[i], evidences[i], position
                    )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

    def _bfs(self, start, goal, walls):
        """
        Perform Breadth-First Search (BFS) to
        find the shortest path from start to goal.

        Arguments:
            start: The starting position (x, y).
            goal: The goal position (x, y).
            walls: The W x H grid of walls.

        Returns:
            A list of positions representing
            the shortest path from start to goal.
        """
        original_pos = {
            (start[0] - 1, start[1]): Directions.WEST,
            (start[0] + 1, start[1]): Directions.EAST,
            (start[0], start[1] - 1): Directions.SOUTH,
            (start[0], start[1] + 1): Directions.NORTH
        }

        fringe = Queue()
        fringe.push([start])
        visited = set()

        while not fringe.isEmpty():
            path = fringe.pop()
            current = path[-1]

            if current == goal:
                return path

            if current not in visited:
                visited.add(current)

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    next_pos = (current[0] + dx, current[1] + dy)
                    if not walls[next_pos[0]][next_pos[1]]:
                        new_path = list(path)
                        new_path.append(next_pos)
                        fringe.push(new_path)

        return []

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Given the current state of the game, returns a legal move for Pacman.

        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        nbr_g, width, height = beliefs.shape
        dist = []
        nearest_g_pos = []

        for g in range(nbr_g):
            if eaten[g]:
                continue

            max_p = 0
            for w in range(width):
                for h in range(height):
                    proba = beliefs[g, w, h]
                    if proba > max_p:
                        max_p = proba
                        g_pos = (w, h)
            nearest_g_pos.append(g_pos)
            dist.append(manhattanDistance(position, g_pos))

        ghost = dist.index(min(dist))
        g_pos = nearest_g_pos[ghost]

        path = self._bfs(position, g_pos, walls)
        if len(path) > 1:
            direction = path[1]
            return {
                (position[0] - 1, position[1]): Directions.WEST,
                (position[0] + 1, position[1]): Directions.EAST,
                (position[0], position[1] - 1): Directions.SOUTH,
                (position[0], position[1] + 1): Directions.NORTH
            }[direction]

        return Directions.STOP

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
