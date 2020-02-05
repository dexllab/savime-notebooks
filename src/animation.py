import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def animate_heat_map(data: np.ndarray):
    fig = plt.figure()
    fig.set_size_inches(3, 2)

    min = np.min(data)
    max = np.max(data)

    def animate(i):
        plt.clf()
        sns.heatmap(data[i], vmin=min, vmax=max)
        fig.canvas.draw()

    plt.tight_layout()

    return animation.FuncAnimation(fig, animate, frames=len(data), interval=400, repeat=False)