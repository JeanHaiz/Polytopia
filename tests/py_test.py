from score_recognition import score_recognition_utils
from tests.src import test_utils


def test_score_recognition_with_2_players():
    image = test_utils.get_score_resource("image_1", )
    assert image is not None
    score = score_recognition_utils.read_scores(image)
    assert len(score) == 2
    assert score[0] == ("KingDadosss", 1525)
    assert score[1] == ("Ruled by you", 1630)


def test_perimeter():
    output = 2*2 + 2*5
    assert output == 14
