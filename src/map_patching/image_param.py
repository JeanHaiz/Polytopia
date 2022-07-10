import numpy as np

from typing import List
from typing import Tuple

from dataclasses import dataclass

from map_patching.corner_orientation import CornerOrientation


@dataclass
class ImageParam:
    filename: str
    cloud_scale: float
    corners: List[Tuple[int, int, int]]

    def getPosition(self) -> Tuple[np.int64, np.int64]:
        # sorted_corners = sorted(self.corners, key=lambda x: x[2])
        # dx = (sorted_corners[0][0] - sorted_corners[2][0]) * 2
        # dy = (sorted_corners[0][1] - sorted_corners[2][1]) * 2
        dx = [c[0] for c in self.corners if c[2] == CornerOrientation.LEFT.value][0]
        dy = [c[1] for c in self.corners if c[2] == CornerOrientation.TOP.value][0]
        return (np.int64(dx), np.int64(dy))

    def getSize(self) -> Tuple[np.int64, np.int64]:
        # sorted_corners = sorted(self.corners, key=lambda x: x[2])
        # TODO: add is_vertical
        if True:  # is_vertical
            # d_h = sorted_corners[1][1] - sorted_corners[0][1]
            # d_w = max(sorted_corners[3][0] - sorted_corners[0][0], sorted_corners[0][0] - sorted_corners[2][0]) * 2
            d_w = [c[0] for c in self.corners if c[2] == CornerOrientation.RIGHT.value][0] - \
                [c[0] for c in self.corners if c[2] == CornerOrientation.LEFT.value][0]
            d_h = [c[1] for c in self.corners if c[2] == CornerOrientation.BOTTOM.value][0] - \
                [c[1] for c in self.corners if c[2] == CornerOrientation.TOP.value][0]
            return (np.int64(d_w), np.int64(d_h))
        else:
            d_h = max(self.corners[2][1] - self.corners[0][1], self.corners[1][1] - self.corners[2][1]) * 2
            d_w = self.corners[3][0] - self.corners[2][0]
            return (np.int64(d_w), np.int64(d_h))
        return None

    def __repr__(self) -> str:
        return str((self.filename, self.cloud_scale, self.corners))
