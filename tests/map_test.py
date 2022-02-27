import cv2
import pytest

from map_patching import map_patching_utils
from tests import test_utils


@pytest.mark.asyncio
async def test_map_patchin_with_1_map():
    output_file_path, filename = await map_patching_utils.patch_partial_maps(
        "general", ["4c01f257-5a2a-4b62-ba86-d15a3b26cec6"], 400, None)

    output = cv2.imread(output_file_path)
    assert output.shape == (1303, 2143, 3)


@pytest.mark.asyncio
async def test_turn_recognition_t0():
    image = test_utils.get_map_resource("image_0")
    assert image is not None
    turn = map_patching_utils.get_turn(image)
    assert turn == "0"


@pytest.mark.asyncio
async def test_turn_recognition_t1():
    image = test_utils.get_map_resource("image_1")
    assert image is not None
    turn = map_patching_utils.get_turn(image)
    assert turn == "1"


@pytest.mark.asyncio
async def test_turn_recognition_t5():
    image = test_utils.get_map_resource("image_5")
    assert image is not None
    turn = map_patching_utils.get_turn(image)
    assert turn == "5"


@pytest.mark.asyncio
async def test_turn_recognition_t7():
    image = test_utils.get_map_resource("image_7")
    assert image is not None
    turn = map_patching_utils.get_turn(image)
    assert turn == "7"


@pytest.mark.asyncio
async def test_turn_recognition_t8():
    image = test_utils.get_map_resource("image_8")
    assert image is not None
    turn = map_patching_utils.get_turn(image)
    assert turn == "8"


@pytest.mark.asyncio
async def test_turn_recognition_t9():
    image = test_utils.get_map_resource("image_9")
    assert image is not None
    turn = map_patching_utils.get_turn(image)
    assert turn == "9"


@pytest.mark.asyncio
async def test_turn_recognition_t10():
    image = test_utils.get_map_resource("image_10")
    assert image is not None
    turn = map_patching_utils.get_turn(image)
    assert turn == "10"


@pytest.mark.asyncio
async def test_turn_recognition_t11():
    image = test_utils.get_map_resource("image_11")
    assert image is not None
    turn = map_patching_utils.get_turn(image)
    assert turn == "11"


@pytest.mark.asyncio
async def test_turn_recognition_t12():
    image = test_utils.get_map_resource("image_12")
    assert image is not None
    turn = map_patching_utils.get_turn(image)
    assert turn == "12"


@pytest.mark.asyncio
async def test_turn_recognition_t13():
    image = test_utils.get_map_resource("image_13")
    assert image is not None
    turn = map_patching_utils.get_turn(image)
    assert turn == "13"


@pytest.mark.asyncio
async def test_turn_recognition_t14():
    image = test_utils.get_map_resource("image_14")
    assert image is not None
    turn = map_patching_utils.get_turn(image)
    assert turn == "14"


@pytest.mark.asyncio
async def test_turn_recognition_t15():
    image = test_utils.get_map_resource("image_15")
    assert image is not None
    turn = map_patching_utils.get_turn(image)
    assert turn == "15"
