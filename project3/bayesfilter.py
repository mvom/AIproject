import numpy as np

from pacman_module.game import Agent, Directions, manhattanDistance


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls, position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

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
        ghost = self.ghost
        trans_model = np.zeros((width, height, width, height))

        fear_value = {"fearless": 1, "afraid": 2, "terrified": 8}.get(ghost, 0)

        for w in range(width):
            for h in range(height):
                # adjacent cells of (w, h)
                next_step = [(w - 1, h), (w + 1, h), (w, h - 1), (w, h + 1)]

                # distance between Pacman and the original case (w, h)
                current_dist = manhattanDistance((w, h), position)

                for i, j in next_step:
                    if walls[w][h] or walls[i][j]:
                        continue

                    else:
                        next_dist = manhattanDistance((i, j), position)

                        value = fear_value if next_dist > current_dist else 1
                        trans_model[i, j, w, h] = value

                # normalization of the matrix along the last two dimensions
                norm = np.sum(trans_model[:, :, w, h])
                if norm:
                    trans_model[:, :, w, h] /= norm

        return trans_model

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """
        # create matrix based off manhattanDistance
        W = walls.width
        H = walls.height
        o = np.empty((W, H))
        for i in range(W):
            for j in range(H):
                if manhattanDistance(position, (i, j)) == evidence:
                    o[i][j] = 0.375  # value of binom(n,p)=np
                elif manhattanDistance(position, (i, j)) == evidence - 1:
                    o[i][j] = 0.250  # value of binom(n,p)=np-1
                elif manhattanDistance(position, (i, j)) == evidence - 2:
                    o[i][j] = 0.0625
                elif manhattanDistance(position, (i, j)) == evidence + 1:
                    o[i][j] = 0.250
                elif manhattanDistance(position, (i, j)) == evidence + 2:
                    o[i][j] = 0.0625
                else:
                    o[i][j] = 0
        return o

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

        trans_matrix = self.transition_matrix(walls, position)
        obser_matrix = self.observation_matrix(walls, evidence, position)

        new_belief = np.zeros((width, height))
        m = np.zeros((width, height))

        for w in range(width):
            for h in range(height):
                m[w][h] = np.sum(trans_matrix[w, h, :, :] * belief)
                new_belief[w, h] = m[w, h] * obser_matrix[w, h]

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
                    walls, beliefs[i], evidences[i], position)

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

    def Heuristic(self, direction, walls, beliefs, eaten, position):

        if direction == str(Directions.NORTH):
            next_pos = (position[0], position[1]+1)
        if direction == str(Directions.EAST):
            next_pos = (position[0]+1, position[1])
        if direction == str(Directions.SOUTH):
            next_pos = (position[0], position[1]-1)
        if direction == str(Directions.WEST):
            next_pos = (position[0]-1, position[1])

        current_best_val = 0
        if walls[next_pos[0]][next_pos[1]]:
            return -1

        # get sum of probabilities across all ghosts
        beliefs = np.nan_to_num(beliefs)
        beliefs_sum = np.sum(beliefs, axis=0)

        for w in range(walls.width):
            for h in range(walls.height):
                # heuristic as probability over distance
                temp_pos = (w, h)
                temp_distance = manhattanDistance(next_pos, temp_pos)
                value = beliefs_sum[w][h] / (temp_distance*10+1)
                # return the maximum value
                if value > current_best_val:
                    current_best_val = value

        return current_best_val

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        current_best_val = 0
        current_best_dir = Directions.STOP

        # set of actions
        directions = (Directions.NORTH, Directions.EAST,
                      Directions.SOUTH, Directions.WEST)

        for direction in directions:
            value = self.Heuristic(
                str(direction), walls, beliefs, eaten, position)
            if value > current_best_val:
                current_best_val = value
                current_best_dir = direction
        return current_best_dir

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
