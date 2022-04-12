import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Fixing random state for reproducibility
np.random.seed(42)

precomputed_frames = 100
total_frames = 130

# Define "offline" data
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

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(231, title="Plot 1", xlim=(0, 20), ylim=(0, 50))
ax2 = fig.add_subplot(232, title="Plot 2", ylim=(0, np.max(heights)+20))
ax3 = fig.add_subplot(233, title="Plot 3")
ax4 = fig.add_subplot(234, title="Plot 4", ylim=(-1.2, 1.2))
ax5 = fig.add_subplot(235, title="Plot 5", xlim=(-4, 4), ylim=(0, 130))
ax6 = fig.add_subplot(236, title="Plot 6", xlim=(0.7, 1.3), ylim=(0.7, 1.3))
plt.tight_layout()

line1, = ax1.plot([], [])
barcollection = ax2.bar(labels, heights[0], color=colors)
wedges, pie_lbs, pie_pct = ax3.pie(heights[0], colors=colors, labels=labels, autopct='%1.1f%%')
line2, = ax4.plot([], [])
_, _, hist_container = ax5.hist([], bins=hist_bins, ec="black")
hist_container = list(hist_container)
scatter, = ax6.plot([], [], marker='o', color='green', alpha=0.5, linestyle='')

# Allocate space for "live" data
n_sine, n_hist, n_scat = 0.3, 3, 1
data_sine = np.full(shape=(total_frames+1), fill_value=np.nan)
data_hist = np.full(shape=((total_frames+1)*n_hist), fill_value=np.nan)
data_scat = np.full(shape=(2, total_frames*n_scat), fill_value=np.nan)

def animate_artist(k):
    i = k if k < (precomputed_frames-1) else (precomputed_frames-1)
    
    # Plot 1
    line1.set_data(x1[:i], y1[:i])

    # Plot 2
    for j, b in enumerate(barcollection):
        b.set_height(heights[i][j])

    # Plot 3
    cumsum_heights = np.insert(np.cumsum(heights[i]), 0, 0)
    h_min, h_max = np.min(cumsum_heights), np.max(cumsum_heights)
    theta = (cumsum_heights - h_min)*360/(h_max-h_min)
    for j, (w, p, l) in enumerate(zip(wedges, pie_pct, pie_lbs)):
        w.set_theta1(theta[j])
        w.set_theta2(theta[j+1])
        radians = np.radians((theta[j+1] - theta[j])/2 + theta[j])
        p.set_x(0.7*np.cos((radians)))
        p.set_y(0.7*np.sin((radians)))
        p.set_text(f"{heights[i][j]/h_max*100 :.1f}%")
        l.set_x(1.2*np.cos((radians)))
        l.set_y(1.2*np.sin((radians)))
    
    # Simulate data feed by appending new data
    data_sine[k] = n_sine*k
    data_hist[n_hist*k:n_hist*(k+1)] = np.random.randn(n_hist)    
    data_scat[:, n_scat*k:n_scat*(k+1)] = np.random.default_rng().lognormal(mean=0, sigma=0.1, size=(2,n_scat))
    
    # Plot 4
    ax4.set_xlim(0, np.nanmax(data_sine)+1)
    line2.set_data(data_sine, np.sin(data_sine))
    
    # Plot 5
    n, _ = np.histogram(data_hist[~np.isnan(data_hist)], bins=hist_bins)
    # for count, rect in zip(n, hist_container.patches):
    for count, rect in zip(n, hist_container):
        rect.set_height(count)

    # Plot 6
    scatter.set_data([data_scat[0, :], data_scat[1, :]])

    return line1, 
    # return (line1, barcollection.patches[0], barcollection.patches[1], barcollection.patches[2], barcollection.patches[3], wedges[0], wedges[1], wedges[2], wedges[3],  line2, hist_container[0], hist_container[1], hist_container[2], hist_container[3], hist_container[4], hist_container[5], hist_container[6], hist_container[7], hist_container[8], hist_container[9], hist_container[10], scatter)


anim = animation.FuncAnimation(fig, animate_artist, frames=total_frames, interval=40, blit=True)
anim.save('./generated_assets/animation2.gif', writer='imagemagick')
plt.show()
