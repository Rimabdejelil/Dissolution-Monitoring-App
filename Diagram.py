import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def add_box(ax, text, xy, boxstyle, facecolor, edgecolor='black'):
    box = FancyBboxPatch(
        xy, 2, 1, boxstyle=boxstyle, facecolor=facecolor, edgecolor=edgecolor, mutation_aspect=2.5
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + 1, xy[1] + 0.5, text, ha='center', va='center', fontsize=10
    )

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Current Manual Process
add_box(ax, 'Manual observation\n[Eye Icon]', (1, 6), 'round,pad=0.3', 'lightblue')
add_box(ax, 'Manual timing and\nrecording\n[Stopwatch Icon]', (1, 4), 'round,pad=0.3', 'lightblue')
add_box(ax, 'Manual data logging\n[Notebook Icon]', (1, 2), 'round,pad=0.3', 'lightblue')

# Desired Automated Process
add_box(ax, 'Real-time image\ncapture\n[Camera Icon]', (8, 6), 'round,pad=0.3', 'lightgreen')
add_box(ax, 'AI-based detection of\ndissolution\n[AI Brain Icon]', (8, 4), 'round,pad=0.3', 'lightgreen')
add_box(ax, 'Real-time prediction\nof dissolution time\n[Graph with clock Icon]', (8, 2), 'round,pad=0.3', 'lightgreen')
add_box(ax, 'Automated data\nlogging\n[Cloud Database Icon]', (8, 0), 'round,pad=0.3', 'lightgreen')

# Arrows
arrowprops = dict(facecolor='black', edgecolor='black', arrowstyle='->')
ax.annotate('', xy=(3, 6.5), xytext=(7, 6.5), arrowprops=arrowprops)
ax.annotate('', xy=(3, 4.5), xytext=(7, 4.5), arrowprops=arrowprops)
ax.annotate('', xy=(3, 2.5), xytext=(7, 2.5), arrowprops=arrowprops)

plt.show()
