import numpy as np

from tqdm import tqdm
from utils import MovingArea
from optimizer import speedProjection
from viztools import plot_animation


class CrowdSolver:
    """Solver for crowd simulation problem"""

    def __init__(
        self,
        area_config: MovingArea,
        max_time: int,
        time_step_number: int,
        initial_positions: np.ndarray,
    ) -> None:
        self._max_time = max_time
        self._time_step_number = time_step_number
        self._time_step = max_time / time_step_number
        self._area_config = area_config
        self._people_number = len(initial_positions)
        self._positions = [initial_positions]

    def solve(self) -> None:
        """Main solver function which compute trajectories"""
        optimizer = speedProjection(moving_zone=self._area_config)
        for _ in tqdm(range(self._time_step_number)):
            positions = np.array(self._positions[-1])
            optimizer.update_position(positions)
            natural_speed = my_area.spontaneous_speeds(positions)
            admissible_speed = optimizer.get_admissible_speed(
                natural_speed=natural_speed, time_step=self._time_step
            )
            self._positions.append(
                self._positions[-1] + admissible_speed * self._time_step
            )

    @property
    def result(self) -> np.ndarray:
        """Get the result from computation"""
        return np.array(self._positions)


if __name__ == "__main__":
    ZONE_WIDTH = 800
    ZONE_HEIGHT = 400
    EXIT_CENTER = np.array([300, ZONE_HEIGHT // 2])  # center of the exit
    EXIT_RADIUS = np.array([10, 20])  # radius of the exit
    PEOPLE_RADIUS = 4
    PEOPLE_NUMBER = 150
    PEOPLE_SPEED = 6

    my_area = MovingArea(
        width=ZONE_WIDTH,
        height=ZONE_HEIGHT,
        moving_speed=PEOPLE_SPEED,
        people_number=PEOPLE_NUMBER,
        people_radius=PEOPLE_RADIUS,
        exit_center=EXIT_CENTER,
        exit_radius=EXIT_RADIUS,
    )
    my_area.initialize_obstacles()
    initial_positions = my_area.initialize_positions(5, True)
    crowd = CrowdSolver(
        area_config=my_area,
        max_time=140,
        time_step_number=40,
        initial_positions=initial_positions,
    )
    crowd.solve()
    trajectories = crowd.result
    plot_animation(trajectories, my_area)
