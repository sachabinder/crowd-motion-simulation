import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import multiprocessing as mp

from tqdm import tqdm
from utils import MovingArea
from optimizer import speedProjection
from viztools import plot_animation, plot_trajectories


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
            natural_speed = self._area_config.spontaneous_speeds(positions)
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


def solve_and_plot(
    zone_width: int = 800,
    zone_height: int = 400,
    exit_center_width: int = None,
    exit_center_height: int = None,
    exit_radius_width: int = 10,
    exit_radius_height: int = 40,
    people_radius: int = 15,
    people_number: int = 20,
    people_speed: int = 6,
    time_step_number: int = 40,
    max_time_step: int = 120,
    random_initial_position: bool = True,
    save_solution: bool = False,
    file_name: str = "data.pkl",
) -> None:
    """Solve the problem and plot the solution"""

    # initial setup of params
    if not exit_center_width:
        exit_center_width = zone_width // 2

    if not exit_center_height:
        exit_center_height = zone_height // 2

    if not random_initial_position:
        people_number = 50  # should fit to the initial positoin pkl file

    exit_center = np.array(
        [exit_center_width, exit_center_height]
    )  # center of the exit
    exit_radius = np.array(
        [exit_radius_width, exit_radius_height]
    )  # radius of the exit

    result = {
        "zone_width": zone_width,
        "zone_height": zone_height,
        "people_radius": people_radius,
        "people_speed": people_speed,
        "people_number": people_number,
        "time_step_number": time_step_number,
        "max_time_step": max_time_step,
        "exit_radius_width": exit_radius[0],
        "exit_radius_height": exit_radius[1],
        "exit_position_width": exit_center[0],
        "exit_position_height": exit_center[1],
    }

    my_area = MovingArea(
        width=zone_width,
        height=zone_height,
        moving_speed=people_speed,
        people_number=people_number,
        people_radius=people_radius,
        exit_center=exit_center,
        exit_radius=exit_radius,
    )

    result["obstacles"] = my_area.initialize_obstacles()
    result["initial_position"] = my_area.initialize_positions(
        marge_safety=5, random=random_initial_position
    )

    crowd = CrowdSolver(
        area_config=my_area,
        max_time=max_time_step,
        time_step_number=time_step_number,
        initial_positions=result["initial_position"],
    )
    crowd.solve()
    trajectories = crowd.result

    if save_solution:
        with open(file_name, "wb") as filehandler:
            pkl.dump(result, filehandler)
            print(f"[i] file saved at {file_name}")

    plot_animation(trajectories, my_area)
    plot_trajectories(trajectories, my_area)


def read_amd_plot_solution(file_path: str):
    with open(file_path, "rb") as f:
        solution = pkl.load(f)
    my_area = MovingArea(
        width=solution["zone_width"],
        height=solution["zone_height"],
        moving_speed=solution["people_speed"],
        people_number=solution["people_number"],
        people_radius=solution["people_radius"],
        exit_center=np.array(
            [solution["exit_position_width"], solution["exit_position_height"]]
        ),
        exit_radius=np.array(
            [solution["exit_radius_width"], solution["exit_radius_height"]]
        ),
    )
    my_area.initialize_obstacles()
    plot_animation(solution["trajectories"], my_area)


if __name__ == "__main__":
    solve_and_plot()
