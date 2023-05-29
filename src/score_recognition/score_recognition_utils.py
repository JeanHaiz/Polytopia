import re
import os
import cv2
import pytesseract

import numpy as np
from difflib import SequenceMatcher
from typing import Callable
from typing import Union
from typing import Tuple
from typing import Dict
from typing import List

from common.logger_utils import logger
from common import image_utils
from common.image_operation import ImageOp

from score_recognition.score_recognition_error import RecognitionException
from database.database_client import get_database_client

DEBUG = int(os.getenv("POLYTOPIA_DEBUG", 0))

database_client = get_database_client()


def score_recognition_request(
        process_uuid: str,
        score_requirement_id: str,
        channel_id: int,
        channel_name: str,
        author_name: str,
        filename: str,
        callback: Union[Callable[[str, str], None], Callable[[str, int, str], None], Callable[[str, str, str], None]]
):
    image = image_utils.load_image(channel_name, filename, ImageOp.SCORE_INPUT)
    
    if image is None:
        raise RecognitionException("SCORE RECOGNITION - IMAGE NOT FOUND: %s, %s" % (channel_name, filename))
    
    scores = __recognise_scores(image)
    
    if scores is None:
        raise RecognitionException("SCORE RECOGNITION - SCORE NOT RECOGNISED: %s, %s" % (channel_name, filename))
    
    turn = database_client.get_last_turn(channel_id) or 0
    
    game_players = database_client.get_game_players(channel_id)
    matching, remaining_scores = __find_matching(game_players, scores, author_name)
    
    for player_uuid, player_name, player_score in matching:
        database_client.set_player_game_name(player_uuid, player_name)
        database_client.add_score(channel_id, player_uuid, player_score, turn)
    
    for player_name, player_score in remaining_scores:
        if player_name is not None and player_name != '':
            player_uuid = database_client.add_missing_player(player_name, channel_id)
        else:
            player_uuid = None
        database_client.add_score(channel_id, player_uuid, player_score, turn)
    
    callback(
        process_uuid,
        channel_id,
        score_requirement_id
    )
    
    print("score recognition done, callback sent", flush=True)


def __recognise_scores(image):
    clear = __clear_noise_optimised(image)
    logger.debug("read image scores")
    image_reading = __read(clear)
    logger.debug("image reading: %s" % image_reading)
    image_text = image_reading.split('\n')
    if DEBUG:
        print("image text", image_text)
    scores = [__read_line(t) for t in image_text if "score" in t]
    logger.debug(scores)
    if DEBUG:
        print("scores", scores)
    return scores


def __read(image, config=''):
    return pytesseract.image_to_string(image, config=config)


def __crop(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    min_line_length = 500
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 2, threshold=200, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=10)
    heights = sorted([lines[i][0][1] for i in range(len(lines))])
    return image[heights[0]:heights[-1], :]


def __read_line(line):
    print("line", line)
    line = re.sub(r"(\\[a-zA-Z0-9]*)", "", line)
    line = re.sub(r"[^a-zA-Z0-9,:]", "", line)
    print("re-line", line)
    s1 = line.split(",")
    
    s1 = [s1_i for s1_i in s1 if s1_i != '' and len(s1_i) > 1]
    
    if len(s1) >= 2:
        if "Unknownruler" in s1[0]:
            player = "Unknown ruler"
        elif "Ruledbyyou" in s1[0]:
            player = "Ruled by you"
        else:
            player = s1[0][len("Ruledby"):]
        
        if player is not None:
            player = re.sub(r"[^a-zA-Z0-9 ]", "", player)
        
        s2 = "".join(s1[1:]).split(":")
        if len(s2) >= 2:
            score = int(s2[1].split("points")[0].replace(",", ""))
        else:
            print("s2 error", line)
            return
    else:
        print("s1 error", line)
        return
    return player, score


def __clear_noise_optimised(image):
    value = (np.array(0.3 * image[:, :, 2] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 0])).astype(np.uint8)
    return cv2.threshold(value, 170, 255, cv2.THRESH_BINARY_INV)[1]


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
