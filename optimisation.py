import numpy as np
import cvxpy as cp

from typing import List
from utils import RectangleObstacle, MovingArea


def directions_between_people(positions: np.ndarray) -> np.ndarray:
    """Copute the (not normalized) vector direction that separate two distinct 
    people for each people of the problem.
    Return a square antisymetric matrix that represent for each people (column), the distance to others.
    """
    positions_matrix = np.array([positions] * len(positions))
    directions = positions_matrix - positions_matrix.transpose(1, 0, 2)
    return directions


def directions_to_walls(
    positions: np.ndarray, moving_zone_width: int, moving_zone_height: int
) -> np.ndarray:
    """Compute the (not normalized) vector direction to each wall 
    of the moving zone
                    
             (0,0)      top wall
                *-----------------------+
      left wall |        Moving         | right wall
                |         zone          |
                +-----------------------*
                         bottom      (widht, height)
    """
    directions_to_top_wall = np.zeros_like(positions)
    directions_to_bottom_wall = np.zeros_like(positions)
    directions_to_left_wall = np.zeros_like(positions)
    directions_to_right_wall = np.zeros_like(positions)

    directions_to_top_wall[:, 1] = -positions[:, 1]
    directions_to_bottom_wall[:, 1] = moving_zone_height - positions[:, 1]
    directions_to_left_wall[:, 0] = -positions[:, 0]
    directions_to_right_wall[:, 0] = moving_zone_width - positions[:, 0]

    return np.array(
        [
            directions_to_top_wall,
            directions_to_bottom_wall,
            directions_to_left_wall,
            directions_to_right_wall,
        ]
    )


def directions_to_obstacles(
    positions: np.ndarray, obstacles: List[RectangleObstacle]
) -> np.ndarray:
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
    for obstacle in obstacles:
        obstacle_directions = np.zeros_like(positions)
        obstacle_directions[:, 0] = (
            (positions[:, 0] <= obstacle.x_min) * obstacle.x_min
            + (positions[:, 0] >= obstacle.x_max) * obstacle.x_max
            + (obstacle.x_min < positions[:, 0])
            * (positions[:, 0] < obstacle.x_max)
            * positions[:, 0]
            - positions[:, 0]
        )
        obstacle_directions[:, 1] = (
            (positions[:, 1] <= obstacle.y_min) * obstacle.y_min
            + (positions[:, 1] >= obstacle.y_max) * obstacle.y_max
            + (obstacle.y_min < positions[:, 1])
            * (positions[:, 1] < obstacle.y_max)
            * positions[:, 1]
            - positions[:, 1]
        )
        directions.append(obstacle_directions)
    return np.array(directions)


def distances(directions: np.ndarray) -> np.ndarray:
    """Distances between two point is the norm of the
    (not normalized) vector direction between them. By default here,
    it is the Euclidian norm.
    """
    return np.linalg.norm(directions, axis=2)


def gradients_of_distances(directions: np.ndarray) -> np.ndarray:
    my_distances = distances(directions)
    gradients = directions / my_distances[:, :, np.newaxis]
    return np.nan_to_num(gradients, nan=0)


def get_admissible_speed(
    natural_speed: np.ndarray,
    positions: np.ndarray,
    moving_zone: MovingArea,
    time_step: float,
) -> np.ndarray:
    people_directions = directions_between_people(positions)
    people_gradients = gradients_of_distances(people_directions)
    people_distances = distances(people_directions)

    walls_directions = directions_to_walls(
        positions=positions,
        moving_zone_width=moving_zone.width,
        moving_zone_height=moving_zone.height,
    )
    walls_gradients = gradients_of_distances(walls_directions)
    walls_distances = distances(walls_directions)

    obstacles_directions = directions_to_obstacles(positions, moving_zone.obstacles)
    obstacles_gradients = gradients_of_distances(obstacles_directions)
    obstacles_distances = distances(obstacles_directions)

    # Optimization problem formulation
    v = cp.Variable(natural_speed.shape)
    objective = cp.Minimize(cp.sum_squares(natural_speed - v))
    constraints = (
        [
            (people_distances[i] + time_step * people_gradients[i] @ v[i] >= 0)
            for i in range(people_directions.shape[0])
        ]
        + [
            (walls_distances[i] + time_step * walls_gradients[i] @ v[i] >= 0)
            for i in range(walls_directions.shape[0])
        ]
        + [
            (obstacles_distances[i] + time_step * obstacles_gradients[i] @ v[i] >= 0)
            for i in range(obstacles_directions.shape[0])
        ]
    )
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return v.value


if __name__ == "__main__":
    """&
    TEST DEV ZONE
    """
    from utils import MovingArea

    WIDTH = 60
    HEIGHT = 30
    EXIT_CENTER = np.array([WIDTH // 2, HEIGHT // 2])  # center of the exit
    EXIT_RADIUS = np.array([3, 2])  # radius of the exit

    my_area = MovingArea(
        width=WIDTH,
        height=HEIGHT,
        moving_speed=1,
        people_radius=1,
        exit_center=EXIT_CENTER,
        exit_radius=EXIT_RADIUS,
    )
    my_area.initialize_obstacles()
    spont_speed = my_area.spontaneous_speeds()
    initial_positions = np.array([[5, 5], [5, 10], [5, 15], [5, 20], [5, 25]])
    specific_spont_speed = spont_speed[initial_positions[:, 0], initial_positions[:, 1]]
    speed = get_admissible_speed(
        natural_speed=specific_spont_speed,
        positions=initial_positions,
        moving_zone=my_area,
        time_step=0.2,
    )
    print(speed)
