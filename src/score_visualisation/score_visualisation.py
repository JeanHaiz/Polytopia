import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero

from common import image_utils
from database.database_client import DatabaseClient


def plot_scores(
        database_client: DatabaseClient,
        scores: pd.DataFrame,
        channel_id: str,
        channel_name: str,
        author_id: str):

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
    return filepath, filename
    # return image_utils.load_attachment(file_path, "Score visualisation")


def augment_scores(scores: pd.DataFrame):
    for player in scores['polytopia_player_name'].drop_duplicates():
        player_scores = scores[scores['polytopia_player_name'] == player]
        player_scores.sort_values(by="turn", ascending=False)
        scores.loc[scores['polytopia_player_name'] == player, "delta"] = player_scores["score"].diff()
    # scores.delta = pd.to_numeric(scores.delta, errors='coerce')
    return scores


def print_scores(scores: pd.DataFrame):
    scores = augment_scores(scores)
    return __print_scores(scores)
    # return augment_scores(scores).to_string(index=False)


def print_player_scores(scores: pd.DataFrame, player):
    scores = augment_scores(scores)
    player_scores = scores[scores['polytopia_player_name'] == player]
    return __print_scores(player_scores)


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
