from typing import Tuple
from matplotlib.patches import Rectangle

import numpy as np
import matplotlib.pyplot as plt

SAFETY_RADIUS = 3  # should implement adaptative raduis


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
    obstacles and border of the area.
    """

    def __init__(
        self,
        width: int,
        height: int,
        moving_speed: float,
        people_radius: int,
        exit_center: np.ndarray,
        exit_radius: np.ndarray,
    ) -> None:
        """
        :param width: width of the moving zone
        :param height: height of the moving zone
        :param moving_speed: speed of individuals (constant here)
        :param people_radius: represent space of people body
        :param exit_center: np.array([width_center, height_center])
        :param exit_radius: np.array([width_exit_radius, height_exit_radius])

        ------------------- widht -----------------
        |                           |-----|
        |                           |-----|
        |
        height            exit_center >
        |
        |                           |-----|
        |                           |-----|
        ---------------------------------------------
        """
        self.width = width
        self.height = height
        self.moving_speed = moving_speed
        self.people_radius = people_radius
        self.exit_area = RectangleObstacle(
            exit_center - exit_radius, exit_center + exit_radius
        )
        self.moving_zone = np.ones((self.width, self.height))
        self.obstacles = []

    def initialize_obstacles(self):
        """Add good obstacles according to the exit area defined by the user."""
        up_wall = RectangleObstacle(
            (self.exit_area.x_min, 0),
            (self.exit_area.x_max, self.exit_area.y_min - self.people_radius),
        )
        down_wall = RectangleObstacle(
            (self.exit_area.x_min, self.exit_area.y_max + self.people_radius),
            (self.exit_area.x_max, self.height),
        )
        self.obstacles.append(up_wall)
        self.obstacles.append(down_wall)

    def spontaneous_speeds(self) -> np.ndarray:
        """Computation of the spontaneous speed of individuals taking into account
        walls, the exit zone and the radius of individuals.
        """
        speeds = np.zeros((self.width, self.height, 2))
        # zone where speed is computed
        for i in range(self.people_radius, self.exit_area.x_min - self.people_radius):
            for j in range(self.people_radius, self.exit_area.y_min):
                direction_vector = np.array(
                    [self.exit_area.x_min - i, self.exit_area.y_min - j]
                )
                speeds[i, j] = (
                    self.moving_speed
                    * direction_vector
                    / np.linalg.norm(direction_vector)
                )
            for k in range(self.exit_area.y_max, self.height - self.people_radius):
                direction_vector = np.array(
                    [self.exit_area.x_min - i, self.exit_area.y_min - k]
                )
                speeds[i, k] = (
                    self.moving_speed
                    * direction_vector
                    / np.linalg.norm(direction_vector)
                )

        speeds[
            self.people_radius + self.exit_area.x_max : self.width - self.people_radius,
            self.people_radius : self.height - self.people_radius,
        ] = np.array([self.moving_speed, 0])
        speeds[
            self.people_radius : self.exit_area.x_max + self.people_radius,
            self.exit_area.y_min : self.exit_area.y_max,
        ] = np.array([self.moving_speed, 0])
        return speeds

    def plot(self, fig: plt.figure) -> None:
        """Plot area and obstacles."""
        ax = fig.add_subplot()
        base_rect = Rectangle(
            xy=(0, 0), width=self.width, height=self.height, color="black", fill=False
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
        """Plote the speed vector field."""
        ax = fig.add_subplot()
        x, y = np.meshgrid(
            np.linspace(0, self.width, self.width),
            np.linspace(0, self.height, self.height),
        )
        speeds = self.spontaneous_speeds()
        ax.quiver(x, y, np.transpose(speeds[:, :, 0]), np.transpose(speeds[:, :, 1]))


if __name__ == "__main__":
    """&
    TEST DEV ZONE
    """

    WIDTH = 60
    HEIGHT = 30
    EXIT_CENTER = np.array([WIDTH // 2, HEIGHT // 2])  # center of the exit
    EXIT_RADIUS = np.array([3, 1])  # radius of the exit
    my_area = MovingArea(
        width=WIDTH,
        height=HEIGHT,
        moving_speed=1,
        people_radius=1,
        exit_center=EXIT_CENTER,
        exit_radius=EXIT_RADIUS,
    )
    my_area.initialize_obstacles()
    fig = plt.figure(figsize=(10, 5))
    my_area.plot_speed(fig)
    my_area.plot(fig)
    plt.axis("equal")
    plt.axis("off")

    plt.show()
