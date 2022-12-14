import numpy as np
import cvxpy as cp

from utils import MovingArea


class speedProjection:
    """Resolve the optimization problem to compute the projection
    on admissible speed set
    """

    def __init__(self, instant_positions: np.ndarray, moving_zone: MovingArea) -> None:
        self._positions = instant_positions
        self._people_number = len(instant_positions)
        self._moving_zone = moving_zone

    def directions_between_people(self) -> np.ndarray:
        """Copute the (not normalized) vector direction that 
        separate two distinct people for each people of the problem.
        Return a square antisymetric matrix that represent for 
        each people (column), the distance to others.
        """
        positions_matrix = np.array([self._positions] * len(self._people_number))
        directions = positions_matrix - positions_matrix.transpose(1, 0, 2)
        return directions[np.triu_indices(len(positions_matrix), k=1)]

    def directions_to_walls(self) -> np.ndarray:
        """Compute the (not normalized) vector direction to each wall 
        of the moving zone
                        
                (0,0)      top wall
                    *-----------------------+
        left wall |        Moving         | right wall
                    |         zone          |
                    +-----------------------*
                            bottom      (widht, height)
        """
        directions_to_top_wall = np.zeros_like(self._positions)
        directions_to_bottom_wall = np.zeros_like(self._positions)
        directions_to_left_wall = np.zeros_like(self._positions)
        directions_to_right_wall = np.zeros_like(self._positions)

        directions_to_top_wall[:, 1] = -self._positions[:, 1]
        directions_to_bottom_wall[:, 1] = (
            self._moving_zone.height - self._positions[:, 1]
        )
        directions_to_left_wall[:, 0] = -self._positions[:, 0]
        directions_to_right_wall[:, 0] = self._moving_zone.width - self._positions[:, 0]

        return np.array(
            [
                directions_to_top_wall,
                directions_to_bottom_wall,
                directions_to_left_wall,
                directions_to_right_wall,
            ]
        )

    def directions_to_obstacles(self) -> np.ndarray:
        """Compute the (not normalized) vector direction to each obstacles 
        of the moving zone. This direction is computing according to the orthogonal projection
        of the position on the obstacle. To do that, we separate an obstacle in 8 zones.

            Zone 2  |       Zone 3          | Zone 4
                    |                       |
            --------+-----------------------|--------
            Zone 1  |        Obstacle       | Zone 5
                    |       (Rectangle)     |
            --------+-----------------------+--------
            Zone 8  |       Zone 7          | Zone 6
                    |                       |
        """
        directions = []
        for obstacle in self._moving_zone.obstacles:
            obstacle_directions = np.zeros_like(self._positions)
            obstacle_directions[:, 0] = (
                (self._positions[:, 0] <= obstacle.x_min) * obstacle.x_min
                + (self._positions[:, 0] >= obstacle.x_max) * obstacle.x_max
                + (obstacle.x_min < self._positions[:, 0])
                * (self._positions[:, 0] < obstacle.x_max)
                * self._positions[:, 0]
                - self._positions[:, 0]
            )
            obstacle_directions[:, 1] = (
                (self._positions[:, 1] <= obstacle.y_min) * obstacle.y_min
                + (self._positions[:, 1] >= obstacle.y_max) * obstacle.y_max
                + (obstacle.y_min < self._positions[:, 1])
                * (self._positions[:, 1] < obstacle.y_max)
                * self._positions[:, 1]
                - self._positions[:, 1]
            )
            directions.append(obstacle_directions)
        return np.array(directions)

    def distances(self, directions: np.ndarray) -> np.ndarray:
        """Distances between two point is the norm of the
        (not normalized) vector direction between them. By default here,
        it is the Euclidian norm.
        """
        return np.linalg.norm(directions, axis=2)

    def gradients_of_distances(self, directions: np.ndarray) -> np.ndarray:
        my_distances = self.distances(directions)
        gradients = directions / my_distances[:, :, np.newaxis]
        return np.nan_to_num(gradients, nan=0)

    def get_admissible_speed(
        self, natural_speed: np.ndarray, time_step: float,
    ) -> np.ndarray:
        people_directions = self.directions_between_people()
        people_gradients = self.gradients_of_distances(people_directions)
        people_distances = np.linalg.norm(people_directions, axis=2)

        walls_directions = self.directions_to_walls()
        walls_gradients = self.gradients_of_distances(walls_directions)
        walls_distances = self.distances(walls_directions)

        obstacles_directions = self.directions_to_obstacles()
        obstacles_gradients = self.gradients_of_distances(obstacles_directions)
        obstacles_distances = self.distances(obstacles_directions)

        # Optimization problem formulation
        v = cp.Variable(natural_speed.shape)
        objective = cp.Minimize(cp.sum_squares(natural_speed - v))
        constraints = (
            [
                (people_distances[i] + time_step * people_gradients[i] @ v[i] >= 0)
                for i in range(people_directions.shape[0])
            ]
            + [
                (walls_distances[i] + time_step * walls_gradients[i] @ v[j] >= 0)
                for i in range(walls_directions.shape[0])
            ]
            + [
                (
                    obstacles_distances[i] + time_step * obstacles_gradients[i] @ v[i]
                    >= 0
                )
                for i in range(obstacles_directions.shape[0])
            ]
        )
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return v.value


if __name__ == "__main__":
    initial_positions = np.array([[5, 5], [5, 10], [5, 15], [5, 20], [5, 25]])
    eij = directions_between_people(initial_positions)
    N = len(initial_positions)
    M = np.zeros((N * (N - 1) // 2, N))
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            M[k, i] = -1
            M[k, j] = +1
            k += 1
    grads = M[:, :, np.newaxis] * eij[:, np.newaxis]
    print(grads)

