import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

# Fixing random state for reproducibility
np.random.seed(42)

precomputed_frames = 100
total_frames = 130

# Define precomputed data
x1 = np.linspace(start=0, stop=20, num=precomputed_frames)
y1 = 0.1*(x1 + 3*np.random.random(size=(len(x1))))**2
heights= np.array([[4+2*i for i,j in enumerate(range(precomputed_frames))],
                   [30+0.3*i for i,j in enumerate(range(precomputed_frames))],
                   [4+1*i for i,j in enumerate(range(precomputed_frames))],
                   [40-0.2*i for i,j in enumerate(range(precomputed_frames))]
                   ], dtype=object).T

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
labels = ['L1', 'L2', 'L3', 'L4']
hist_bins = np.linspace(-4, 4, 12)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,5))
xlim_list = [(0, 20), (-1, 4), (0, 1), (0, 40), (-4, 4), (0.7, 1.3)]
ylim_list = [(0, 50), (0, 220), (0, 1), (-1.2, 1.2), (0, 130), (0.7, 1.3)]
title_list = [f"Plot {i+1}" for i in range(6)]
plt.tight_layout()

# Define title and axis limits
for (ax, title, xlim, ylim) in zip(axes.flatten(), title_list, xlim_list, ylim_list):
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Allocate space for "live" data
n_sine, n_hist, n_scat = 0.3, 3, 1
data_sine = np.full(shape=(total_frames+1), fill_value=np.nan)
data_hist = np.full(shape=((total_frames+1)*n_hist), fill_value=np.nan)
data_scat = np.full(shape=(2, total_frames*n_scat), fill_value=np.nan)

def animate_celluloid(i):
    i = k if k < (precomputed_frames-1) else (precomputed_frames-1)

    # Simulate data feed by appending new data
    data_sine[k] = n_sine*(k+1)
    data_hist[n_hist*k:n_hist*(k+1)] = np.random.randn(n_hist)
    data_scat[:, n_scat*k:n_scat*(k+1)] = np.random.default_rng().lognormal(mean=0, sigma=0.1, size=(2,n_scat))

    # Dynamically adjust the x-axis limits for Plot 4
    # Spoiler: This doesn't work for celluloid
    xlim_list[3] = (0, np.nanmax(data_sine))
    axes[1][0].set_xlim(xlim_list[3])

    # Plot 1
    axes[0][0].plot(x1[:i], y1[:i], color='blue')
    # Plot 2
    axes[0][1].bar(labels, heights[i], color=colors)
    # Plot 3
    axes[0][2].pie(heights[i], colors=colors, labels=labels, autopct='%1.1f%%')
    # Plot 4
    axes[1][0].plot(data_sine, np.sin(data_sine))
    # Plot 5
    axes[1][1].hist(data_hist, bins=hist_bins, ec='black', fc="blue")
    # Plot 6
    axes[1][2].scatter(x=data_scat[0, :], y=data_scat[1, :], c="green", alpha=0.5)


camera = Camera(fig)
for k in range(total_frames):
    animate_celluloid(k)
    camera.snap()

anim = camera.animate(interval=50, blit=False)
anim.save('./generated_assets/animation3.gif', dpi=100)
