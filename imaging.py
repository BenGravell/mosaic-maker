"""Functions for handling images."""

from io import BytesIO

import numpy as np
import PIL


def create_final_img(img_lib, sol, tgt_shape):
    img_lib_x, img_lib_y = img_lib.shape[1:3]
    patch_idxs = np.array(np.unravel_index(sol[0], shape=tgt_shape)).T.tolist()
    final_im_arr = np.zeros((tgt_shape[1] * img_lib_x, tgt_shape[0] * img_lib_y, 3), dtype=np.uint8)
    for sol_src, sol_tgt in zip(*sol):
        j, i = patch_idxs[sol_src]

        x1 = i * img_lib_x
        x2 = (i + 1) * img_lib_x
        y1 = j * img_lib_y
        y2 = (j + 1) * img_lib_y

        final_im_arr[x1:x2, y1:y2] = img_lib[sol_tgt]
    return PIL.Image.fromarray(final_im_arr)


def save_image_to_bytes(image, format="PNG"):
    buf = BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()
