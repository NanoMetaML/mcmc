"""
File: collect.py
Author: Blake Wilson 
Email: wilso692@purdue.edu

Description:
    Tests nanomcmc package with various platforms.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib
from matplotlib.transforms import Affine2D


if __name__ == "__main__":
    matplotlib.use("svg")

    plt.tight_layout()

    x = torch.load("./test/gifs/sampled_vecs.npy")

    w = 12
    h = 16
    n = w * h

    temperatures = [0.01, 0.1, 1.0, 10, 100, 1000]

    for i in range(len(x)):

        fig, ax = plt.subplots(1, 1, figsize=(3, 4))
        im = ax.imshow(x[i].reshape(w, h).cpu().transpose(0, 1), cmap="afmhot")
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        interval = 1

        plt.savefig(f"./test/gifs/last_sample_{i}.png")

    def animate(i):

        im = ax.imshow(x[i].reshape(w, h).cpu().transpose(0, 1), cmap="afmhot")
        ax.set_title(f"Temperature = {temperatures[i]}")

        return (im,)

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=(len(x) - 1), interval=50
    )
    ani.save("./test/gifs/animation.gif", writer="imagemagick", fps=2)
