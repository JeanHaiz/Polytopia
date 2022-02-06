import cv2
import pytest

# from common import image_utils
from map_patching import map_patching_utils


@pytest.mark.asyncio
async def test_score_recognition_with_2_players():
    output_file_path, filename = await map_patching_utils.patch_partial_maps(
        "general", ["4c01f257-5a2a-4b62-ba86-d15a3b26cec6"], 400, None)

    output = cv2.imread(output_file_path)
    # output = image_utils.load_attachment(output_file_path, filename)
    assert output.shape == (1303, 2143, 3)
