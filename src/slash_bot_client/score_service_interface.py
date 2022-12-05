import numpy as np
import pandas as pd

from typing import List
from typing import Dict
from typing import Tuple
from typing import Optional
from difflib import SequenceMatcher
from interactions import File
from interactions import Embed
from interactions import Channel
from interactions import Message
from interactions import CommandContext

from common import image_utils
from common.image_operation import ImageOp  # TODO change

from database.database_client import DatabaseClient


def is_score_recognition_request(reactions, attachment, filename):
    # TODO: complete with image analysis heuristics
    return "📈" in [r.emoji for r in reactions]


async def get_scores(database_client: DatabaseClient, ctx: CommandContext) -> None:
    scores: pd.DataFrame = database_client.get_channel_scores(ctx.channel_id)
    if scores is not None and len(scores[scores['turn'] != -1]) > 0:
        scores = scores[scores['turn'] != -1]
        channel = await ctx.get_channel()
        filepath, filename = score_visualisation.plot_scores(
            database_client, scores, channel.id, channel.name, ctx.author.id)
        with open(filepath, "rb") as fh:
            attachment = File(fp=fh, filename=filename + ".png")
            image_utils.load_attachment(filepath, "Score visualisation")
            await channel.send(files=attachment, content="score recognition")
        
        score_text = score_visualisation.print_scores(scores)
        # await ctx.send(score_text)
        embed = Embed(title='Game scores', description=score_text)
        await ctx.send(embeds=embed)
    else:
        await ctx.send("No score found")


async def process_score_recognition(
        database_client: DatabaseClient,
        channel: Channel,
        message: Message) -> None:
    if len(message.attachments) == 1:
        output = ""
        for i, attachment in enumerate(message.attachments):
            if is_score_recognition_request(message.reactions, attachment, "filename"):
                filename = await prepare_attachment(
                    database_client, channel, message, attachment, i, ImageOp.SCORE_INPUT)
                score_text = await score_recognition_routine(database_client, message, filename)
                print("score text:", score_text)
                if score_text is not None:
                    if output != "":
                        output += "\n\n"
                    output += score_text
        if len(output) > 0:
            await channel.send(output)
        print("output", output)
    elif len(message.attachments) > 1:
        await message.reply("Only one image per message is currently supported.")


async def get_player_scores(database_client: DatabaseClient, ctx: CommandContext, player: str) -> None:
    scores = database_client.get_channel_scores(ctx.channel.id)
    if scores is not None and len(scores[scores['turn'] != -1]) > 0:
        if player is not None and player not in scores["polytopia_player_name"].unique():
            players = scores["polytopia_player_name"].unique()
            await ctx.send("Player %s not recognised. Available players: %s" % (str(player), str(players)))
        else:
            score_text = score_visualisation.print_player_scores(scores, player)
            embed = Embed(title='%s scores' % str(player), description=score_text)
            await ctx.send(embeds=embed)
    else:
        await ctx.send("No score found for player %s" % str(player))


async def score_recognition_routine(
        database_client: DatabaseClient,
        message: Message,
        filename: str) -> Optional[str]:
    channel = await message.get_channel()
    channel_name = channel.name
    channel_id = channel.id
    
    image_check = await image_utils.get_or_fetch_image_check(
        database_client,
        lambda i: message.attachments[i].download(),
        channel_name,
        message.id,
        filename,
        ImageOp.SCORE_INPUT
    )
    if not image_check:
        return None
    
    scores = __read_scores(channel_name, filename, ImageOp.SCORE_INPUT)
    turn = database_client.get_last_turn(channel_id) or 0
    
    if scores is None:
        return None
    
    game_players = database_client.get_game_players(channel_id)
    matching, remaining_scores = __find_matching(game_players, scores, message.author.username)
    
    for player_uuid, player_name, player_score in matching:
        database_client.set_player_game_name(player_uuid, player_name)
        database_client.add_score(channel_id, player_uuid, player_score, turn)
    
    for player_name, player_score in remaining_scores:
        if player_name is not None and player_name != '':
            player_uuid = database_client.add_missing_player(player_name, channel_id)
        else:
            player_uuid = None
        database_client.add_score(channel_id, player_uuid, player_score, turn)
    
    score_text = "Scores for turn %d:\n" % turn
    score_text += "\n".join([(s[0] or "Unknown ruler") + ": " + str(s[1]) for s in scores])
    return score_text


def __find_matching(
        game_players: List[dict],
        scores: List[Tuple[str, int]],
        author_name: str) -> Tuple[list, list]:
    def get_game_player_name(game_player: Dict) -> str:
        return game_player["polytopia_player_name"] or game_player["discord_player_name"] or ""
    
    score_players = [
        (None, s[1]) if s[0] == "Unknown ruler"
        else ((author_name, s[1]) if s[0] == "Ruled by you" else s)
        for s in scores]
    
    similarity = np.zeros((len(score_players), len(game_players)))
    
    for i in range(len(score_players)):
        for j in range(len(game_players)):
            print("pre-text matching", score_players[i], game_players[j])
            print("text matching", score_players[i][0], get_game_player_name(game_players[j]))
            if score_players[i][0] is None:
                sim_ij = 0
            else:
                sim_ij = SequenceMatcher(None, score_players[i][0], get_game_player_name(game_players[j])).ratio()
            similarity[i, j] = sim_ij
    
    output = []
    while len(similarity) > 0 and len(similarity[0]) > 0:
        index = np.where(similarity == np.amax(similarity))
        index_max = index[0][0], index[1][0]
        
        if similarity[index_max[0], index_max[1]] == 0:
            if len(game_players) != 1 or len(score_players) != 1:
                break
        
        output.append((
            game_players[index_max[1]]["game_player_uuid"],
            score_players[index_max[0]][0],
            score_players[index_max[0]][1]))
        game_players.pop(index_max[1])
        score_players.pop(index_max[0])
        
        similarity = np.delete(np.delete(similarity, index_max[0], 0), index_max[1], 1)
    
    # Handle the case where there are more than 1 Unknown ruler
    
    return output, score_players


def __read_scores(
        channel_name: str,
        filename: str,
        operation: ImageOp
) -> List:
    return []  # TODO complete
    # TODO add callback
