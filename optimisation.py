import numpy as np

# import cvxpy as cp
from typing import List
from utils import RectangleObstacle


def directions_between_people(positions: np.ndarray) -> np.ndarray:
    """Copute the (not normalized) vector direction that separate two distinct people
    """
    positions_matrix = np.array([positions] * len(positions))
    directions_vector_matrix = positions_matrix - positions_matrix.transpose(1, 0, 2)
    directions = directions_vector_matrix[np.triu_indices(len(positions_matrix), k=1)]
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

    return np.concatenate(
        (
            directions_to_top_wall,
            directions_to_bottom_wall,
            directions_to_left_wall,
            directions_to_right_wall,
        ),
        axis=0,
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
    return np.concatenate(directions, axis=0)


def distances(directions: np.ndarray) -> np.ndarray:
    """Distances between two point is the norm of the
    (not normalized) vector direction between them. By default here,
    it is the Euclidian norm.
    """
    return np.linalg.norm(directions, axis=1)


def gradients_of_distances(directions: np.ndarray) -> np.ndarray:
    my_distances = distances(directions)
    return directions / my_distances[:, np.newaxis]


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

    initial_positions = np.array([[5, 5], [5, 10], [5, 15], [5, 20], [5, 25]])
    people_directions = directions_between_people(initial_positions)
    walls_directions = directions_to_walls(
        initial_positions,
        moving_zone_width=my_area.width,
        moving_zone_height=my_area.height,
    )
    obstacles_directions = directions_to_obstacles(initial_positions, my_area.obstacles)
    directions = np.concatenate(
        [people_directions, walls_directions, obstacles_directions]
    )
    my_distances = distances(directions=directions)
    gradients = gradients_of_distances(directions)
