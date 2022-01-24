import os

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero

from common import image_utils


def plotScores(scores, channel_name, filename):

    x = list(scores['turn'])
    y = list(scores['score'])

    fig = plt.figure()
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)

    ax.scatter(x, y)
    for i in range(len(x)):
        ax.text(x[i], y[i], i, color="red", fontsize=14)

    ax.tick_params(axis='x', colors='#FFFFFF')
    ax.tick_params(axis='y', colors='#FFFFFF')

    ax.set_xticks(np.arange(np.min(x), np.max(x) + 1))

    plt.tight_layout()

    parent_path, file_path = image_utils.get_plt_path(channel_name, filename)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)

    plt.savefig(file_path)
    plt.show()
    return image_utils.load_attachment(file_path, "Score recognition")
