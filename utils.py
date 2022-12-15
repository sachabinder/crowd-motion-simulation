import numpy as np
import matplotlib.pyplot as plt

from matplotlib import axes
from typing import Tuple
from matplotlib.patches import Rectangle, Circle
from random import randint
from sklearn.preprocessing import normalize

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
        people_number: int,
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
        self.people_number = people_number
        self.exit_area = RectangleObstacle(
            exit_center - exit_radius, exit_center + exit_radius
        )
        self.moving_zone = np.ones((self.width, self.height))
        self.obstacles = []
        self.initial_positions = []

    def initialize_obstacles(self) -> None:
        """Add good obstacles according to the exit area defined by the user."""
        up_wall = RectangleObstacle(
            (self.exit_area.x_min, 0), (self.exit_area.x_max, self.exit_area.y_min),
        )
        down_wall = RectangleObstacle(
            (self.exit_area.x_min, self.exit_area.y_max),
            (self.exit_area.x_max, self.height),
        )
        self.obstacles.append(up_wall)
        self.obstacles.append(down_wall)

    def initialize_positions(self, marge_safety: int, random: bool) -> np.ndarray:
        """Generate initial postion (randomly) with the constraints to respect
        """
        positions = []
        x = randint(
            self.people_radius + marge_safety,
            self.exit_area.x_min - self.people_radius - marge_safety,
        )
        y = randint(
            self.people_radius + marge_safety,
            self.height - self.people_radius - marge_safety,
        )
        candidate_position = np.array([x, y])
        positions.append(candidate_position)
        for _ in range(self.people_number - 1):
            while True:
                flag = True
                x = randint(
                    self.people_radius + marge_safety,
                    self.exit_area.x_min - self.people_radius - marge_safety,
                )
                y = randint(
                    self.people_radius + marge_safety,
                    self.height - self.people_radius - marge_safety,
                )
                candidate_position = np.array([x, y])
                for position in positions:
                    if (
                        np.linalg.norm(position - candidate_position)
                        < 2 * self.people_radius + marge_safety
                    ):
                        flag = False
                if flag:
                    break
            positions.append(candidate_position)
        self.initial_positions = np.array(positions)
        return positions

    def spontaneous_speeds(self, positions: np.ndarray) -> np.ndarray:
        """Computation of the spontaneous speed of individuals taking into account
        walls, the exit zone and the radius of individuals.
        """
        speeds = np.zeros_like(positions)
        speeds[:, 0] = (positions[:, 0] <= self.exit_area.x_min - 1) * (
            (
                (positions[:, 1] <= self.exit_area.y_min - 1)
                + (positions[:, 1] >= self.exit_area.y_max)
            )
            * (self.exit_area.x_min - 1 - positions[:, 0])
            + (positions[:, 1] > self.exit_area.y_min - 1)
            * (positions[:, 1] < self.exit_area.y_max)
        ) + (positions[:, 0] > self.exit_area.x_min - 1) * (
            self.width - positions[:, 0]
        )

        speeds[:, 1] = (positions[:, 0] <= self.exit_area.x_min - 1) * (
            (
                (positions[:, 1] <= self.exit_area.y_min - 1)
                + (positions[:, 1] >= self.exit_area.y_max)
            )
            * (self.exit_area.y_min + 1 - positions[:, 1])
        )
        return normalize(speeds, axis=1) * self.moving_speed

    def plot(self, ax: axes) -> None:
        """Plot area and obstacles."""
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

    def plot_speed(self, ax: axes) -> None:
        """Plot the speed vector field."""
        x, y = np.meshgrid(
            np.linspace(0, self.width, self.width),
            np.linspace(0, self.height, self.height),
        )
        speeds = np.zeros((self.width, self.height, 2))
        for i in range(self.width):
            for j in range(self.height):
                speeds[i, j] = self.spontaneous_speeds(np.array([[i, j]]))[0]
        ax.quiver(x, y, np.transpose(speeds[:, :, 0]), np.transpose(speeds[:, :, 1]))

    def plot_initial_positions(self, ax: axes) -> None:
        """Plot people disposed initially"""
        for position in self.initial_positions:
            circle = Circle(position, self.people_radius, color="r")
            ax.add_patch(circle)


if __name__ == "__main__":
    """&
    TEST DEV ZONE
    """

    WIDTH = 40
    HEIGHT = 20
    EXIT_CENTER = np.array([2 * WIDTH // 3, HEIGHT // 2])  # center of the exit
    EXIT_RADIUS = np.array([1, 2])  # radius of the exit
    my_area = MovingArea(
        width=WIDTH,
        height=HEIGHT,
        moving_speed=1,
        people_radius=1,
        people_number=15,
        exit_center=EXIT_CENTER,
        exit_radius=EXIT_RADIUS,
    )
    my_area.initialize_obstacles()
    my_area.initialize_positions(marge_safety=0, random=True)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()

    my_area.plot_speed(ax)
    my_area.plot(ax)
    # my_area.plot_initial_positions(ax)
    plt.axis("equal")
    plt.axis("off")

    plt.show()
