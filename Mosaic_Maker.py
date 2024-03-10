"""Streamlit app for creating image mosaics."""

import numpy as np
import PIL
import streamlit as st

import data_loading as dl
import distance
import assignment
import imaging
import utils


TARGET_RESOLUTION_OPTIONS = utils.logspace_2_and_3(2, 8)


def load_dataset(dataset_name: str):
    """Wrapper around dl.load_source_images to provide streamlit caching."""
    if st.session_state.get("dataset_name") == dataset_name:
        st.toast(f"Dataset `{dataset_name}` already loaded :smile:")
        return
    with st.spinner("Loading dataset..."):
        st.session_state.imgs = dl.load_source_images(dataset_name)
        st.session_state.dataset_name = dataset_name


def get_target_image():
    st.header("Select Target Image")
    use_default_image = st.toggle("Use Default Image")
    if use_default_image:
        default_image_name = st.selectbox(
            "Default Image", options=["umbrellas", "bouquet"], format_func=lambda x: x.title()
        )
        return PIL.Image.open(f"sample_images/{default_image_name}.jpg").convert("RGB")

    st.session_state.uploaded_file = st.file_uploader(
        "Upload target image", type=[".jpg", ".png"], label_visibility="collapsed"
    )
    uploaded_file = st.session_state.get("uploaded_file")
    if uploaded_file is None:
        st.info("Target image not uploaded yet.")
        return None

    return PIL.Image.open(uploaded_file).convert("RGB")


def create_mosaic(imgs, tgt_img_src, tgt_res, assignment_algorithm):
    if imgs is None:
        raise ValueError("Dataset not loaded yet.")

    img_lib = np.array(imgs)
    Y = np.mean(img_lib, axis=(1, 2))

    tgt_img_orig_size = tgt_img_src.size
    tgt_img_new_size = tuple(np.round(np.array(tgt_img_orig_size) * (tgt_res / max(tgt_img_orig_size))).astype(int))
    N = np.prod(tgt_img_new_size)

    if len(imgs) < N:
        raise ValueError(
            "Fewer library images than pixels in target image. Reduce the output resolution or provide a larger library"
            " of source images."
        )

    tgt_img = tgt_img_src.resize(tgt_img_new_size)
    tgt_arr = np.array(tgt_img)[:, :, 0:3]
    X = np.reshape(tgt_arr, (N, 3), order="F")

    with st.spinner("Calculating distance matrix..."):
        D = distance.compute_distance_matrix(X, Y)
    with st.spinner("Solving assignment problem..."):
        sol = assignment.compute_assignment(D, assignment_algorithm)
    with st.spinner("Assembling final mosaic image..."):
        final_img = imaging.create_final_img(img_lib, sol, tgt_img_new_size)
    return final_img


def display_image_result(img, name):
    st.subheader(name.title())
    if img is None:
        st.info(f"{name} not generated yet.")
        return

    st.image(img)

    downloads_as_png = st.toggle(
        "Download as PNG",
        help="Filesizes will be larger and download buttons will take longer to render.",
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
        file_name=f"target.{ext}",
        mime=mime,
        use_container_width=True,
    )


def display_image_results(tgt_img_src, final_img):
    st.header("Results")
    cols = st.columns(2)
    with cols[0]:
        display_image_result(tgt_img_src, "Target image")
    with cols[1]:
        display_image_result(final_img, "Mosaic image")


def main():
    with st.sidebar:
        st.header("Options")
        dataset_name = st.selectbox("Source Image Dataset Name", ["CIFAR100", "CIFAR10"])
        tgt_res = st.select_slider(
            "Target Resolution",
            options=TARGET_RESOLUTION_OPTIONS,
            value=64,
            help="Number of source image patches to use along the largest target image dimension.",
        )
        assignment_algorithm_description_map = {
            "greedy_random": "Greedy Random (fast, low quality)",
            "jonker_volgenant": "Jonker-Volgenant (slow, high quality)",
        }
        assignment_algorithm = st.selectbox(
            "Assignment Algorithm",
            options=["greedy_random", "jonker_volgenant"],
            format_func=lambda x: assignment_algorithm_description_map[x],
        )

        if assignment_algorithm == "jonker_volgenant" and tgt_res > 64:
            st.warning(
                "The Jonker-Volgenant assignment algorithm is not recommended for target resolutions greater than 64."
                " You may experience slow or hung mosaic generation."
            )

        st.divider()

        st.header("Actions")
        do_load_dataset = st.button("Load Source Image Dataset", use_container_width=True)
        do_generate_mosaic = st.button("Generate Mosaic", use_container_width=True)

    if dataset_name != st.session_state.get("dataset_name"):
        st.session_state.imgs = None
    if do_load_dataset:
        load_dataset(dataset_name)
    imgs = st.session_state.get("imgs")

    tgt_img_src = get_target_image()

    if do_generate_mosaic:
        try:
            st.session_state.final_img = create_mosaic(imgs, tgt_img_src, tgt_res, assignment_algorithm)
        except Exception as exc:
            st.warning("Could not generate mosaic due to the following exception.")
            st.exception(exc)
    final_img = st.session_state.get("final_img")

    display_image_results(tgt_img_src, final_img)


if __name__ == "__main__":
    main()
