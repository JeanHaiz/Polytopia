import os

import numpy as np
import pandas as pd

from typing import Callable

import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero

from common import image_utils
from database.database_client import get_database_client

database_client = get_database_client()


def plot_scores(
        process_uuid: str,
        channel_id: int,
        channel_name: str,
        author_id: str,
        callback: Callable[[str, int, str], None]
):
    scores: pd.DataFrame = database_client.get_channel_scores(channel_id)
    filename = database_client.add_visualisation(channel_id, author_id)
    database_client.add_visualisation_scores(filename, scores)

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

    if len(x) > 0 and len(y) > 0:
        ax.scatter(x, y)
        for i in range(len(x)):
            ax.text(x[i], y[i], i, color="red", fontsize=14)

        ax.set_xticks(np.arange(np.min(x), np.max(x) + 1))

    ax.tick_params(axis='x', colors='#FFFFFF')
    ax.tick_params(axis='y', colors='#FFFFFF')

    plt.legend()
    plt.tight_layout()

    parent_path, filepath = image_utils.get_plt_path(channel_name, filename)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)

    plt.savefig(filepath)
    # plt.show()
    # return filepath, filename

    callback(
        process_uuid,
        channel_id,
        filename
    )
    # return image_utils.load_attachment(file_path, "Score visualisation")


def __print_scores(scores):
    print(scores)
    scores = scores.drop(columns="score_uuid")
    score_list = scores.to_numpy().tolist()
    player_width = np.max([len(d) if type(d) == str else 0 for data in score_list for d in data])
    print("player_width", player_width)

    def width(i):
        return player_width if i == 0 else 5

    def align(i, item):
        if i == 0:
            if item is None:
                item = "  ?"
            return item.ljust(width(i), ' ')
        else:
            return str(item).rjust(width(i), ' ')

    scores.sort_values(by="turn")

    header = ['Player', 'Turn', 'Score', 'Delta']
    s = ['   '.join([str(item).ljust(width(i), ' ') for i, item in enumerate(header)])]

    for data in score_list:
        s.append('   '.join([align(i, item) for i, item in enumerate(data)]))
    d = '```' + '\n'.join(s) + '```'
    return d
