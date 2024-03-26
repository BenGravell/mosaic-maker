"""Functions for handling images."""

from io import BytesIO

import numpy as np
import PIL.Image  # type: ignore[import]
from PIL.Image import Image  # type: ignore[import]

from type_defs import AssignmentSolution, ArrU8


def create_final_img(img_lib_arr: ArrU8, sol: AssignmentSolution, tgt_shape: tuple[int, int]) -> Image:
    """Create the final mosaic image.

    Arguments:
    img_lib_arr: Array of shape [N, W, H, C] where N is the number of images, W is the width of the images in pixels, H is the height of the images in pixels, and C is the number of channels.
    sol: Assignment solution, a tuple of [source index list, target index list].
    tgt_shape: A tuple representing the shape of the final mosaic in terms of the number of tiles.
    """
    img_lib_x, img_lib_y = img_lib_arr.shape[1:3]
    patch_idxs = np.array(np.unravel_index(sol[0], shape=tgt_shape)).T.tolist()
    final_im_arr = np.zeros((tgt_shape[1] * img_lib_x, tgt_shape[0] * img_lib_y, 3), dtype=np.uint8)
    for sol_src, sol_tgt in zip(*sol):
        j, i = patch_idxs[sol_src]

        x1 = i * img_lib_x
        x2 = (i + 1) * img_lib_x
        y1 = j * img_lib_y
        y2 = (j + 1) * img_lib_y

        final_im_arr[x1:x2, y1:y2] = img_lib_arr[sol_tgt]
    return PIL.Image.fromarray(final_im_arr)


def save_image_to_bytes(image: Image, format: str = "PNG") -> bytes:
    """Save an image to in-memory bytes."""
    buf = BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()
