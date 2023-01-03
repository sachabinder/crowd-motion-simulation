import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

from utils import MovingArea


def plot_animation(trajectories: np.ndarray, area: MovingArea) -> None:
    fig = plt.figure(figsize=(10, 5))
    fig.set_dpi(100)
    fig.set_size_inches(7, 6.5)

    ax = plt.axes(xlim=(0, area.width), ylim=(0, area.height))
    plt.axis("equal")
    plt.axis("off")
    patches = [
        plt.Circle(trajectories[0, i], area.people_radius, fc="r")
        for i in range(trajectories.shape[1])
    ]

    def init():
        area.plot(ax)
        for patch in patches:
            ax.add_patch(patch)
        return patches

    def animate(i):
        for j, patch in enumerate(patches):
            patch.center = trajectories[i, j]
        return patches

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(trajectories), interval=200, blit=True
    )
    plt.show()

