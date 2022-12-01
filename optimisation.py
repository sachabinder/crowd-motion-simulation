import numpy as np

# import cvxpy as cp

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
                    
             (0,0)       top wall
                *-----------------------+
      left wall |        Moving         | right wall
                |         zone          |
                +-----------------------*
                        bottom wall     (widht, height)
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


def distance_obstacle(
    positions: np.ndarray, people_radius: int, obstacle: RectangleObstacle
) -> np.ndarray:

    """We have to check in which zone the people are to compute their distance to the obstacle
        Zone 2  |       Zone 3          | Zone 4
                |                       |
        --------+-----------------------|--------
        Zone 1  |        Obstacle       | Zone 5
                |       (Rectangle)     |
        --------+-----------------------+--------
        Zone 8  |       Zone 7          | Zone 6
                |                       |
    """

    distances = []
    for k in range(len(positions)):

        x = positions[k][0]
        y = positions[k][1]

        # Check Zone 1 and 5
        if obstacle.x_min <= x <= obstacle.x_max:
            if y <= obstacle.y_min:  # Zone 1
                distances.append(abs(y - obstacle.y_min))
            elif y >= obstacle.y_max:  # Zone 5
                distances.append(abs(y - obstacle.y_max))
            else:
                return "error: a person is in an obstacle..."

        # Check Zone 1 and 5
        elif obstacle.y_min <= y <= obstacle.y_max:
            if x <= obstacle.x_min:  # Zone 3
                distances.append(abs(x - obstacle.x_min))
            elif x >= obstacle.x_max:  # Zone 7
                distances.append(abs(x - obstacle.x_max))
            else:
                return "error: a person is in an obstacle..."

        # Check Zone 2 and 4
        elif x < obstacle.x_min:
            if y < obstacle.y_min:  # Zone 2
                distances.append(
                    np.linalg.norm(
                        positions[k] - np.array([obstacle.x_min, obstacle.y_min])
                    )
                )
            elif y > obstacle.y_max:  # Zone 4
                distances.append(
                    np.linalg.norm(
                        positions[k] - np.array([obstacle.x_max, obstacle.y_min])
                    )
                )

        # Check Zone 6 and 8
        elif x > obstacle.x_max:
            if y < obstacle.y_min:  # Zone 8
                distances.append(
                    np.linalg.norm(
                        positions[k] - np.array([obstacle.x_max, obstacle.y_min])
                    )
                )
            elif y > obstacle.y_max:  # Zone 6
                distances.append(
                    np.linalg.norm(
                        positions[k] - np.array([obstacle.x_max, obstacle.y_max])
                    )
                )

    return distances

