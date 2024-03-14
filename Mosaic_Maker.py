"""Streamlit app for creating image mosaics."""

import numpy as np
import PIL
from PIL.Image import Image, blend
import streamlit as st

import data_loading as dl
import imaging
import constants
import assignment


def streamlit_setup():
    st.set_page_config(page_title="Mosaic Maker", page_icon="🖼️", layout="wide")


def get_mosaic_options_from_ui():
    out = {}
    out["dataset_name"] = st.selectbox("Source Image Dataset Name", ["CIFAR100", "CIFAR10"])

    cols = st.columns(2)
    with cols[0]:
        out["X_batch_size"] = st.select_slider(
            "Target Image Batch Size",
            options=[100, 500, 1000, 2000],
            value=1000,
            help=(
                "Max number of target image pixels used in the batches used when solving assignment problems. Larger"
                " sizes expose more choices for the assignment, resulting in higher quality mosaics, but increase CPU"
                " and RAM usage."
            ),
        )
    with cols[1]:
        out["Y_batch_size"] = st.select_slider(
            "Source Image Batch Size",
            options=[100, 500, 1000, 2000, 5000, 10000, 20000],
            value=5000,
            help=(
                "Max number of source images used in the batches used when solving assignment problems. Larger sizes"
                " expose more choices for the assignment, resulting in higher quality mosaics, but increase CPU and RAM"
                " usage."
            ),
        )

    out["tgt_res"] = st.select_slider(
        "Mosaic Resolution",
        options=constants.TARGET_RESOLUTION_OPTIONS,
        value=64,
        help="Number of source image patches to use along the largest target image dimension.",
    )

    out["target_img_blend_alpha"] = st.slider(
        "Target Image Blend Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help=(
            'Blend the mosaic image with the (pixelated) target image. This allows you to "cheat" by using color'
            ' information directly from the target image. This will have the effect of "washing out" the colors in the'
            ' small patch images. A setting of 0.0 will yield a "true" mosaic, with no "cheating". A setting of 1.0'
            " will just yield the literal (pixelated) target image. Depending on the mosaic, you may be able to get"
            " away with up to a setting of 0.30 before it starts to become obvious. Most mosaics will look best with a"
            " setting between 0 and 0.20."
        ),
    )

    return out


@st.cache_resource(max_entries=2)
def load_source_images(dataset_name: str) -> list[Image]:
    return np.array(dl.load_source_images(dataset_name))


def load_dataset(dataset_name: str):
    """Wrapper around dl.load_source_images to provide streamlit caching."""
    if st.session_state.get("dataset_name") == dataset_name and st.session_state.get("source_img_arr") is not None:
        return
    with st.spinner("Loading dataset..."):
        st.session_state.source_img_arr = load_source_images(dataset_name)
        st.session_state.dataset_name = dataset_name


def get_target_image():
    use_sample_image = st.toggle("Use Sample Image")
    if use_sample_image:
        image_name = st.selectbox(
            "Sample Image",
            options=["umbrellas", "tulips", "scuba", "abstract", "bouquet"],
            format_func=lambda x: x.title(),
        )
        return PIL.Image.open(f"sample_images/{image_name}.jpg").convert("RGB"), image_name

    st.session_state.uploaded_file = st.file_uploader(
        "Upload target image", type=[".jpg", ".png"], label_visibility="collapsed"
    )
    if (uploaded_file := st.session_state.get("uploaded_file")) is None:
        return None, None

    return PIL.Image.open(uploaded_file).convert("RGB"), uploaded_file.name


def create_mosaic(
    source_img_arr, tgt_img_src, tgt_res, assignment_algorithm, X_batch_size, Y_batch_size, target_img_blend_alpha
):
    Y = np.mean(source_img_arr, axis=(1, 2))

    tgt_img_orig_size = tgt_img_src.size
    tgt_img_new_size = tuple(np.round(np.array(tgt_img_orig_size) * (tgt_res / max(tgt_img_orig_size))).astype(int))
    N = np.prod(tgt_img_new_size)

    if source_img_arr.shape[0] < N:
        raise ValueError(
            "Fewer library images than pixels in target image. Reduce the output resolution or provide a larger library"
            " of source images."
        )

    tgt_img = tgt_img_src.resize(tgt_img_new_size)
    tgt_arr = np.array(tgt_img)
    X = np.reshape(tgt_arr, (N, 3), order="F")

    with st.spinner("Solving assignment problem..."):
        sol = assignment.compute_assignment_batched(X, Y, X_batch_size, Y_batch_size, assignment_algorithm)
    with st.spinner("Assembling final mosaic image..."):
        final_img = imaging.create_final_img(source_img_arr, sol, tgt_img_new_size)
        # Apply blending
        tgt_img_at_final_img_size = tgt_img.resize(final_img.size, resample=PIL.Image.NEAREST)
        final_img = blend(final_img, tgt_img_at_final_img_size, target_img_blend_alpha)
    return final_img


def display_image_result(img, name, filename):
    if img is None:
        st.info(f"{name} not generated yet.")
        return

    # Create a thumbnail to manage the image height, since streamlit does not provide any way to control it...
    thumbnail = img.copy()
    thumbnail.thumbnail((400, 400))
    st.image(thumbnail, use_column_width=False, caption=f"{name} (preview)")

    downloads_as_png = st.toggle(
        "Download as PNG",
        help="PNG filesizes will be larger and download buttons will take longer to render.",
        key=f"download_as_png__{name}",
    )

    format = "JPEG"
    ext = "jpg"
    mime = "image/jpeg"

    if downloads_as_png:
        format = "PNG"
        ext = "png"
        mime = "image/png"

    st.download_button(
        label=f"Download {name.title()} ({format})",
        data=imaging.save_image_to_bytes(img, format=format),
        file_name=f'{filename}_{name.lower().replace(" ", "_")}.{ext}',
        mime=mime,
        use_container_width=True,
    )


def main():
    streamlit_setup()
    target_img_st_column, mosaic_img_st_column = st.columns(2)

    with target_img_st_column:
        st.header("Target Image", anchor=False)
        with st.expander("Options", expanded=True):
            tgt_img_src, tgt_img_filename = get_target_image()

        display_image_result(tgt_img_src, "Target image", filename=tgt_img_filename)

    with mosaic_img_st_column:
        st.header("Mosaic Image", anchor=False)
        with st.expander("Options", expanded=True):
            mosaic_options = get_mosaic_options_from_ui()
            do_generate_mosaic = st.button("Generate Mosaic", use_container_width=True)

        load_dataset(mosaic_options["dataset_name"])
        if (source_img_arr := st.session_state.get("source_img_arr")) is None:
            st.warning("Source images not loaded yet.")
        elif tgt_img_src is None:
            st.warning("Target image not selected yet.")
        else:
            if do_generate_mosaic:
                try:
                    st.session_state.final_img = create_mosaic(
                        source_img_arr,
                        tgt_img_src,
                        mosaic_options["tgt_res"],
                        "jonker_volgenant",
                        mosaic_options["X_batch_size"],
                        mosaic_options["Y_batch_size"],
                        mosaic_options["target_img_blend_alpha"],
                    )
                except Exception as exc:
                    st.warning("Could not generate mosaic due to the following exception.")
                    st.exception(exc)
            final_img = st.session_state.get("final_img")
            display_image_result(final_img, "Mosaic image", filename=tgt_img_filename)


if __name__ == "__main__":
    main()
