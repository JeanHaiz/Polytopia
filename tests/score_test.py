from score_recognition import score_recognition_utils
from tests import test_utils


def test_score_recognition_with_2_players():
    image = test_utils.get_score_resource("image_1")
    assert image is not None
    score = score_recognition_utils.__recognise_scores(image)
    assert len(score) == 2
    assert score[1] == ("KingDadosss", 1525)
    assert score[0] == ("Ruled by you", 1630)


def test_score_recognition_with_4_players():
    image = test_utils.get_score_resource("image_2")
    assert image is not None
    score = score_recognition_utils.__recognise_scores(image)
    assert len(score) == 4
    assert score[0] == ("ZeBiggestPotato", 6285)
    assert score[2] == ("Unknown ruler", 4045)
    assert score[3] == ("Samaterribl", 3315)  # should be Samdterribl
    assert score[1] == ("Ruled by you", 4060)


def test_score_recognition_with_6_players():
    image = test_utils.get_score_resource("image_3")
    assert image is not None
    score = score_recognition_utils.__recognise_scores(image)
    assert len(score) == 6
    assert score[0] == ("Unknown ruler", 1650)
    assert score[1] == ("Unknown ruler", 1495)
    assert score[2] == ("Unknown ruler", 1495)
    assert score[3] == ("Unknown ruler", 1445)
    assert score[4] == ("Ruled by you", 1290)
    assert score[5] == ("Unknown ruler", 1015)


def test_score_recognition_with_4_players_bis():
    image = test_utils.get_score_resource("image_4")
    assert image is not None
    score = score_recognition_utils.__recognise_scores(image)
    assert len(score) == 4
    assert score[0] == ("Player1", 2035)
    assert score[1] == ("Nuupi", 1530)
    assert score[2] == ("Unknown ruler", 1515)
    assert score[3] == ("Npico", 1240)
