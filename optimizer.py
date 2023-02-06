import numpy as np
import cvxpy as cp

from utils import MovingArea


class speedProjection:
    """Resolve the optimization problem to compute the projection
    on admissible speed set
    """

    def __init__(self, moving_zone: MovingArea) -> None:
        self._positions = None
        self._moving_zone = moving_zone
        self._people_gradient_transition_matrix = np.zeros(
            (
                self._moving_zone.people_number
                * (self._moving_zone.people_number - 1)
                // 2,
                self._moving_zone.people_number,
            )
        )
        k = 0
        for i in range(self._moving_zone.people_number):
            for j in range(i + 1, self._moving_zone.people_number):
                self._people_gradient_transition_matrix[k, i] = -1
                self._people_gradient_transition_matrix[k, j] = +1
                k += 1

    def update_position(self, new_positions: np.ndarray):
        self._positions = new_positions

    def directions_between_people(self) -> np.ndarray:
        """Copute the (not normalized) vector direction that
        separate two distinct people for each people of the problem.
        Return a square antisymetric matrix that represent for
        each people (column), the distance to others.
        """
        positions_matrix = np.array([self._positions] * len(self._positions))
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

    def gradients_of_distances(self, directions: np.ndarray) -> np.ndarray:
        my_distances = np.linalg.norm(directions, axis=2)
        gradients = directions / my_distances[:, :, np.newaxis]
        return np.nan_to_num(gradients, nan=0)

    def get_admissible_speed(
        self,
        natural_speed: np.ndarray,
        time_step: float,
    ) -> np.ndarray:
        people_directions = self.directions_between_people()
        people_distances = (
            np.linalg.norm(people_directions, axis=1)
            - 2 * self._moving_zone.people_radius
        )
        people_gradient = (
            self._people_gradient_transition_matrix[:, :, np.newaxis]
            * (people_directions / people_distances[:, np.newaxis])[:, np.newaxis]
        )

        walls_directions = self.directions_to_walls()
        walls_gradients = -self.gradients_of_distances(walls_directions)
        walls_distances = (
            np.linalg.norm(walls_directions, axis=2) - self._moving_zone.people_radius
        )

        obstacles_directions = self.directions_to_obstacles()
        obstacles_gradients = -self.gradients_of_distances(obstacles_directions)
        obstacles_distances = (
            np.linalg.norm(obstacles_directions, axis=2)
            - self._moving_zone.people_radius
        )

        assert (
            people_distances.shape[0] == people_gradient.shape[0]
        ), "Missmatch of dims between people gradient and people distances !"

        # Optimization problem formulation
        v = cp.Variable(natural_speed.shape)
        objective = cp.Minimize(cp.sum_squares(natural_speed - v))
        constraints = (
            [
                people_distances[k]
                + time_step * cp.sum(cp.multiply(people_gradient[k], v))
                >= 0
                for k in range(len(people_distances))
            ]
            + [
                walls_distances[k]
                + time_step * cp.sum(cp.multiply(walls_gradients[k], v), axis=1)
                >= 0
                for k in range(len(walls_distances))
            ]
            + [
                obstacles_distances[k]
                + time_step * cp.sum(cp.multiply(obstacles_gradients[k], v), axis=1)
                >= 0
                for k in range(len(obstacles_distances))
            ]
        )
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        return v.value
