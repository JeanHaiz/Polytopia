import cv2
import pytest
import numpy

from map_patching import map_patching_utils
from map_patching import header_recognition

from tests import test_utils


@pytest.mark.asyncio
async def test_map_patchin_with_1_map() -> None:
    channel_name = "testing-resources"
    files = ["image_1"]
    map_size = 400
    images = await test_utils.prepare_test_images(files, channel_name, map_size)
    assert len(images) == 2
    assert images[0] is not None and type(images[0]) == numpy.ndarray
    assert images[1] is not None and type(images[0]) == numpy.ndarray

    output_file_path, filename, patching_errors = map_patching_utils.patch_partial_maps(
        channel_name, images, files, map_size, 0)

    assert patching_errors == []
    assert filename == "map_patching_debug"
    output = cv2.imread(output_file_path)
    assert output.shape == (1303, 2143, 3)


@pytest.mark.asyncio
async def test_turn_recognition_t0() -> None:
    image = test_utils.get_map_resource("image_0")
    assert image is not None
    turn = header_recognition.get_turn(image)
    assert turn == "0"


@pytest.mark.asyncio
async def test_turn_recognition_t1() -> None:
    image = test_utils.get_map_resource("image_1")
    assert image is not None
    turn = header_recognition.get_turn(image)
    assert turn == "1"


@pytest.mark.asyncio
async def test_turn_recognition_t5() -> None:
    image = test_utils.get_map_resource("image_5")
    assert image is not None
    turn = header_recognition.get_turn(image)
    assert turn == "5"


@pytest.mark.asyncio
async def test_turn_recognition_t7() -> None:
    image = test_utils.get_map_resource("image_7")
    assert image is not None
    turn = header_recognition.get_turn(image)
    assert turn == "7"


@pytest.mark.asyncio
async def test_turn_recognition_t8() -> None:
    image = test_utils.get_map_resource("image_8")
    assert image is not None
    turn = header_recognition.get_turn(image)
    assert turn == "8"


@pytest.mark.asyncio
async def test_turn_recognition_t9() -> None:
    image = test_utils.get_map_resource("image_9")
    assert image is not None
    turn = header_recognition.get_turn(image)
    assert turn == "9"


@pytest.mark.asyncio
async def test_turn_recognition_t10() -> None:
    image = test_utils.get_map_resource("image_10")
    assert image is not None
    turn = header_recognition.get_turn(image)
    assert turn == "10"


@pytest.mark.asyncio
async def test_turn_recognition_t11() -> None:
    image = test_utils.get_map_resource("image_11")
    assert image is not None
    turn = header_recognition.get_turn(image)
    assert turn == "11"


@pytest.mark.asyncio
async def test_turn_recognition_t12() -> None:
    image = test_utils.get_map_resource("image_12")
    assert image is not None
    turn = header_recognition.get_turn(image)
    assert turn == "12"


@pytest.mark.asyncio
async def test_turn_recognition_t13() -> None:
    image = test_utils.get_map_resource("image_13")
    assert image is not None
    turn = header_recognition.get_turn(image)
    assert turn == "13"


@pytest.mark.asyncio
async def test_turn_recognition_t14() -> None:
    image = test_utils.get_map_resource("image_14")
    assert image is not None
    turn = header_recognition.get_turn(image)
    assert turn == "14"


@pytest.mark.asyncio
async def test_turn_recognition_t15() -> None:
    image = test_utils.get_map_resource("image_15")
    assert image is not None
    turn = header_recognition.get_turn(image)
    assert turn == "15"
