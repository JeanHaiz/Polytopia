import os

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero

from common import image_utils


def plotScores(scores, channel_name, filename):

    x = list(scores[scores['polytopia_player_name'].isna()]['turn'])
    y = list(scores[scores['polytopia_player_name'].isna()]['score'])

    fig = plt.figure()
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)

    score_groups = scores.groupby(by="polytopia_player_name").agg({'turn': list, 'score': list})
    print(score_groups)
    for player_name, score_series in score_groups.iterrows():
        player_turns = score_series['turn']
        player_scores = score_series["score"]
        ax.plot(player_turns, player_scores, '-o', label=player_name)

    ax.scatter(x, y)
    for i in range(len(x)):
        ax.text(x[i], y[i], i, color="red", fontsize=14)

    ax.tick_params(axis='x', colors='#FFFFFF')
    ax.tick_params(axis='y', colors='#FFFFFF')

    ax.set_xticks(np.arange(np.min(x), np.max(x) + 1))

    plt.legend()
    plt.tight_layout()

    parent_path, file_path = image_utils.get_plt_path(channel_name, filename)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)

    plt.savefig(file_path)
    plt.show()
    return image_utils.load_attachment(file_path, "Score recognition")
