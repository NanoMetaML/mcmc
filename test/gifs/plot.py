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


if __name__ == "__main__":
    plt.tight_layout()

    x = torch.load("./test/gifs/sampled_vecs.npy")["chosen"][-500:]
    w = 12
    h = 16
    n = w * h

    fig, ax = plt.subplots()
    im = ax.imshow(x[-1].reshape(w, h), cmap="rainbow")
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    interval = 1

    plt.savefig("./test/gifs/last_sample.png")

    def animate(i):
        x[i].reshape(h, w).cpu().numpy()
        im.set_data(x[i].reshape(h, w))
        return (im,)

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=(len(x) - 1), interval=50
    )
    ani.save("./test/gifs/animation.gif", writer="imagemagick", fps=30)
