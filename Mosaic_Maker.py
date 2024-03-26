"""Streamlit app for creating image mosaics."""

import dataclasses

import numpy as np
import PIL.Image  # type: ignore[import]
from PIL.Image import Image, blend  # type: ignore[import]
import streamlit as st  # type: ignore[import]

import data_loading as dl
import imaging
import constants
import assignment
from type_defs import ArrU8


@dataclasses.dataclass
class AppOptions:
    dataset_name: str
    X_batch_size: int
    Y_batch_size: int
    tgt_res: int
    target_img_blend_alpha: float
    assignment_algorithm: str


def streamlit_setup() -> None:
    """Set up Streamlit."""
    st.set_page_config(page_title="Mosaic Maker", page_icon="🖼️", layout="wide")


def get_app_options_from_ui() -> AppOptions:
    """Get app options from the UI."""
    dataset_name = st.selectbox("Source Image Dataset", constants.DATASET_NAME_OPTIONS)

    cols = st.columns(2)
    with cols[0]:
        X_batch_size = st.select_slider(
            "Target Image Batch Size",
            options=constants.X_BATCH_SIZE_OPTIONS,
            value=256,
            help=(
                "Max number of target image pixels used in the batches used when solving assignment problems. Larger"
                " sizes expose more choices for the assignment, resulting in higher quality mosaics, but increase CPU"
                " and RAM usage."
            ),
        )
    with cols[1]:
        Y_batch_size = st.select_slider(
            "Source Image Batch Size",
            options=constants.Y_BATCH_SIZE_OPTIONS,
            value=8192,
            help=(
                "Max number of source images used in the batches used when solving assignment problems. Larger sizes"
                " expose more choices for the assignment, resulting in higher quality mosaics, but increase CPU and RAM"
                " usage."
            ),
        )

    assignment_algorithm = st.selectbox(
        "Assignment Algorithm",
        options=constants.ASSIGNMENT_ALGORITHM_OPTIONS,
        format_func=constants.ASSIGNMENT_ALGORITHM_OPTIONS_FORMAT_FUNC,
        help=(
            "Algorithm for solving the assignment problem for each batch. Jonker-Volgenant returns the best (optimal)"
            " solutions, but runs slower. Greedy-Random returns good (suboptimal) solutions, and runs faster,"
            " especially with a large Source Image Batch Size."
        ),
    )

    tgt_res = st.select_slider(
        "Mosaic Resolution",
        options=constants.TARGET_RESOLUTION_OPTIONS,
        value=96,
        help="Number of source image patches to use along the largest target image dimension.",
    )

    target_img_blend_alpha = st.select_slider(
        "Target Image Blend Alpha",
        options=constants.TARGET_IMG_BLEND_ALPHA_OPTIONS,
        value=0.0,
        help=(
            'Blend the mosaic image with the (pixelated) target image. This allows you to "cheat" by using color'
            ' information directly from the target image. This will have the effect of "washing out" the colors in the'
            ' small patch images. A setting of 0.0 will yield a "true" mosaic, with no "cheating". A setting of 1.0'
            " will just yield the literal (pixelated) target image. Depending on the mosaic, you may be able to get"
            " away with up to a setting of 0.3 before it starts to become obvious. Most mosaics will look best with a"
            " setting between 0 and 0.2."
        ),
    )

    return AppOptions(
        dataset_name=dataset_name,
        X_batch_size=X_batch_size,
        Y_batch_size=Y_batch_size,
        tgt_res=tgt_res,
        target_img_blend_alpha=target_img_blend_alpha,
        assignment_algorithm=assignment_algorithm,
    )


@st.cache_resource(max_entries=2)
def load_source_images(dataset_name: str) -> ArrU8:
    """Wrapper around dl.load_source_images to provide streamlit caching."""
    return np.array(dl.load_source_images(dataset_name))


def load_dataset(dataset_name: str) -> None:
    if st.session_state.get("dataset_name") == dataset_name and st.session_state.get("source_img_arr") is not None:
        return
    st.session_state.source_img_arr = load_source_images(dataset_name)
    st.session_state.dataset_name = dataset_name


def get_target_image() -> tuple[Image | None, str]:
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
        return None, ""

    return PIL.Image.open(uploaded_file).convert("RGB"), uploaded_file.name


def create_mosaic(
    source_img_arr: ArrU8,
    tgt_img_src: Image,
    app_options: AppOptions,
) -> Image:
    Y = np.mean(source_img_arr, axis=(1, 2)).astype(np.int64)

    tgt_img_orig_size = tgt_img_src.size
    tgt_img_new_size = tuple(
        np.round(np.array(tgt_img_orig_size) * (app_options.tgt_res / max(tgt_img_orig_size))).astype(int)
    )
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
        sol = assignment.compute_assignment_batched(
            X, Y, app_options.X_batch_size, app_options.Y_batch_size, app_options.assignment_algorithm
        )
    with st.spinner("Assembling final mosaic image..."):
        final_img = imaging.create_final_img(source_img_arr, sol, tgt_img_new_size)
        # Apply blending
        tgt_img_at_final_img_size = tgt_img.resize(final_img.size, resample=PIL.Image.NEAREST)
        final_img = blend(final_img, tgt_img_at_final_img_size, app_options.target_img_blend_alpha)
    return final_img


def display_image_result(img: Image | None, name: str, filename: str) -> None:
    if img is None:
        st.info(f"{name} not generated yet.")
        return

    # Create a thumbnail to manage the image height, since streamlit does not provide any way to control it...
    thumbnail = img.copy()
    thumbnail.thumbnail((400, 400))
    st.image(thumbnail, use_column_width=False, caption=f"{name} (preview)")

    downloads_as_png = st.toggle(
        "Download as PNG",
        help="PNG filesize will be larger and download button will take longer to render.",
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


def main() -> None:
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
            app_options = get_app_options_from_ui()

        do_generate_mosaic = st.button("Generate Mosaic", use_container_width=True)

        load_dataset(app_options.dataset_name)
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
                        app_options,
                    )
                except Exception as exc:
                    st.warning("Could not generate mosaic due to the following exception.")
                    st.exception(exc)
            final_img = st.session_state.get("final_img")
            display_image_result(final_img, "Mosaic image", filename=tgt_img_filename)


if __name__ == "__main__":
    main()
