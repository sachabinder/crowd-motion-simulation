from typing import Tuple, List
from matplotlib.patches import Rectangle

import numpy as np
import matplotlib.pyplot as plt


class RectangleObstacle:
    """Rectagle obstacles wich one cannot penetrate
    It is defined as follow :

    (x_min, y_min)----------|
        |                       |
        |                       |
        |-------------------(x_max,y_max)
    """

    def __init__(self, point_1: Tuple[int], point_2: Tuple[int]) -> None:
        self.x_min, self.y_min = (
            min(point_1[0], point_2[0]),
            min(point_1[1], point_2[1]),
        )
        self.x_max, self.y_max = (
            max(point_1[0], point_2[0]),
            max(point_1[1], point_2[1]),
        )
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min

    def __str__(self) -> str:
        return (
            "Rectangle obstacle of vertices "
            f"({self.x_min}, {self.y_min}) ({self.x_max}, {self.y_max})"
        )


class MovingArea:
    """Area where one can move. An entity cannot go throught
    obstacles and bordel of the area.
    """

    def __init__(
        self,
        width: int,
        height: int,
        moving_speed: float,
        exit_point: Tuple[int] = None,
        mesure_unit: float = 1,
    ) -> None:
        self._width = width
        self._height = height
        self.moving_speed = moving_speed
        self._mesure_unit = mesure_unit
        self._exit = (
            np.array(exit_point)
            if exit_point
            else np.array((self._width // 2, self._height // 2))
        )
        self.obstacles = []

    def initialize_obstacles(self, obstacles: List[RectangleObstacle]):
        """Check if obstacles are well defined and add them to the area."""
        for obstacle in obstacles:
            assert (
                0 <= obstacle.x_min <= self._width
                and 0 <= obstacle.x_max <= self._width
                and 0 <= obstacle.y_min <= self._height
                and 0 <= obstacle.y_max <= self._height
            ), f"[!] Your {obstacle} is not well defined !"
            self.obstacles.append(obstacle)

    def _is_in_escape_zone(self, position: Tuple[int]) -> bool:
        return position[0] < self._exit[0]

    def spontaneous_speeds(self) -> np.ndarray:
        speeds = np.zeros((self._width, self._height, 2))
        for x in range(self._width):
            for y in range(self._height):
                direction_vector = self._exit - np.array((x, y))
                speeds[x, y] = (
                    (
                        self.moving_speed
                        * direction_vector
                        / np.linalg.norm(direction_vector)
                    )
                    if self._is_in_escape_zone((x, y))
                    else np.array((self.moving_speed, 0))
                )
        return speeds

    def plot(self, fig: plt.figure) -> None:
        """Plot area and obstacles."""
        ax = fig.add_subplot()
        base_rect = Rectangle(
            xy=(0, 0), width=self._width, height=self._height, color="black", fill=False
        )
        ax.add_patch(base_rect)
        for obstacle in self.obstacles:
            obstacle_rect = Rectangle(
                xy=(obstacle.x_min, obstacle.y_min),
                width=obstacle.width,
                height=obstacle.height,
                color="grey",
            )
            ax.add_patch(obstacle_rect)

    def plot_speed(self, fig):
        ax = fig.add_subplot()
        x, y = np.meshgrid(
            np.linspace(0, self._width, self._width),
            np.linspace(0, self._height, self._height),
        )
        speeds = self.spontaneous_speeds()
        ax.quiver(x, y, np.transpose(speeds[:, :, 0]), np.transpose(speeds[:, :, 1]))


if __name__ == "__main__":
    """
    TEST DEV ZONE
    """

    WIDTH = 20
    HEIGHT = 12
    EXIT_RADIUS = 1
    WALL_BORDER = 1

    my_board = MovingArea(WIDTH, HEIGHT, 1)
    obstacles = [
        RectangleObstacle(
            (WIDTH // 2, 0),
            (WIDTH // 2 + WALL_BORDER, HEIGHT // 2 - EXIT_RADIUS),
        ),
        RectangleObstacle(
            (WIDTH // 2, HEIGHT),
            (WIDTH // 2 + WALL_BORDER, HEIGHT // 2 + EXIT_RADIUS),
        ),
    ]
    my_board.initialize_obstacles(obstacles)
    speeds = my_board.spontaneous_speeds()

    fig = plt.figure()

    my_board.plot_speed(fig)
    my_board.plot(fig)
    plt.axis("equal")
    plt.axis("off")
    plt.show()
