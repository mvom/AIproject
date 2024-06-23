from pacman_module.game import Agent
from pacman_module.pacman import Directions

class PacmanAgent(Agent):
    def __init__(self):
        super().__init__()

    def key(self, state):
        return (
            state.getPacmanPosition(),
            state.getNumAgents(),
            state.getFood(),
            state.getGhostPosition(1)
        )

    def h_minimax(self, state, depth, is_maximizing):
        if depth == 0 or state.isWin() or state.isLose():
            return self.heuristic_function(state)

        if is_maximizing:
            bestValue = float("-inf")
            for successor_state, _ in state.generatePacmanSuccessors():
                value = self.h_minimax(successor_state, depth - 1, False)
                bestValue = max(bestValue, value)
            return bestValue
        else:
            bestValue = float("inf")
            for successor_state, _ in state.generateGhostSuccessors(1):
                value = self.h_minimax(successor_state, depth - 1, True)
                bestValue = min(bestValue, value)
            return bestValue

    def heuristic_function(self, state):
        pacman_position = state.getPacmanPosition()
        ghost_positions = [state.getGhostPosition(i) for i in range(1, state.getNumAgents())]

        # Pénaliser les états où Pac-Man est trop proche des fantômes
        min_ghost_distance = min([self.manhattan_distance(pacman_position, ghost_pos) for ghost_pos in ghost_positions])

        if min_ghost_distance < 2:
            return -100  # Pénalité élevée pour éviter les états dangereux

        # Trouver la distance minimale entre Pac-Man et la nourriture
        food_positions = state.getFood().asList()
        if not food_positions:
            return state.getScore()  # Si toute la nourriture a été mangée, évaluez le score

        min_distance = min([self.manhattan_distance(pacman_position, food_pos) for food_pos in food_positions])

        # Reste de la fonction heuristique
        return 1 / min_distance

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_action(self, state):
        current_best_value = float("-inf")
        current_best_action = Directions.STOP

        for successor_state, next_action in state.generatePacmanSuccessors():
            value = self.h_minimax(successor_state, depth=2, is_maximizing=False)
            if value > current_best_value:
                current_best_value = value
                current_best_action = next_action

        return current_best_action
