# Mosaic Maker

Create mosaics from thousands of unique images.

<img src="sample_images/umbrellas.jpg" height="400"> ➡️ <img src="https://i.imgur.com/yLJ4r09.jpg" height="400">

[View the example mosaic on Imgur in full quality](https://imgur.com/gallery/AUaAlhb)

## Running the app

### Run remotely

Open the deployed app at <https://mosaic-maker.streamlit.app/>

### Run locally

Run

```bash
streamlit run Mosaic_Maker.py
```

## Usage guide

1. Load your target image.
2. Adjust the options.
3. Click "Generate Mosaic".
4. Download the generated mosaic image.

![Mosaic maker example app usage screenshot](app_usage_images/mosaic_maker_screenshot.jpg)

## Setup

### Create a Virtual Environment

Create a virtual environment in the `.venv` directory (recommended):

```bash
python -m venv .venv
```

### Activate the Virtual Environment

- **On macOS/Linux:**

  ```bash
  source .venv/bin/activate
  ```

- **On Windows (CMD):**

  ```cmd
  .venv\Scripts\activate.bat
  ```

- **On Windows (PowerShell):**

  ```powershell
  .venv\Scripts\Activate.ps1
  ```

### Install Dependencies

Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

## Development

### Pre-commit

Run

```bash
pre-commit run --all-files
```

to run all pre-commit hooks, including style formatting and unit tests.

### Package management

Update [`requirements.in`](requirements.in) with new direct dependencies.

Then run

```bash
pip-compile requirements.in
```

to update the [`requirements.txt`](requirements.txt) file with all indirect and transitive dependencies.

Then run

```bash
pip install -r requirements.txt
```

to update your virtual environment with the packages.

## Acknowledgements

### Sample Images

- [`sample_images/umbrellas.jpg`](/sample_images/umbrellas.jpg) is an image by [Guy Stevens](https://unsplash.com/@gstevens0884?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) provided under the [Unsplash License](https://unsplash.com/license) on [Unsplash](https://unsplash.com/photos/person-taking-photo-of-assorted-color-umbrellas-dEGu-oCuB1Y?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash).
- [`sample_images/tulips.jpg`](/sample_images/tulips.jpg) is an image by [Kwang Mathurosemontri](https://unsplash.com/@gemini_zucha89?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) provided under the [Unsplash License](https://unsplash.com/license) on [Unsplash](https://unsplash.com/photos/shallow-focus-photography-of-white-and-pink-petaled-flowers-fY1ECB1RCd0?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash).
- [`sample_images/scuba.jpg`](/sample_images/scuba.jpg) is an image by [NEOM](https://unsplash.com/@neom?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) provided under the [Unsplash License](https://unsplash.com/license) on [Unsplash](https://unsplash.com/photos/a-person-swimming-over-a-colorful-coral-reef-eNIGxtOdB10?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash).
- [`sample_images/abstract.jpg`](/sample_images/abstract.jpg) is an image by [Martin Katler](https://unsplash.com/@martinkatler?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) provided under the [Unsplash License](https://unsplash.com/license) on [Unsplash](https://unsplash.com/photos/a-red-white-and-blue-abstract-background-S-Lm2lhayi0?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash).
- [`sample_images/husky.jpg`](/sample_images/husky.jpg) is an image by [Liviu Roman](https://unsplash.com/@liviuroman?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) provided under the [Unsplash License](https://unsplash.com/license) on [Unsplash](https://unsplash.com/photos/close-up-photography-black-and-white-siberian-husky-mNmOgYtwVpQ?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash).
- [`sample_images/peacock.jpg`](/sample_images/husky.jpg) is an image by [Ricardo Frantz](https://unsplash.com/@ricardofrantz?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) provided under the [Unsplash License](https://unsplash.com/license) on [Unsplash](https://unsplash.com/photos/photo-of-blue-and-green-peacock-GvyyGV2uWns?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash).
- [`sample_images/seagull.jpg`](/sample_images/seagull.jpg) is ["See through you"](https://www.flickr.com/photos/jurvetson/99473679/) by [Steve Jurvetson](https://www.flickr.com/photos/jurvetson/) provided under a [CC BY 2.0 License](https://creativecommons.org/licenses/by/2.0/).
- [`sample_images/macaw.jpg`](/sample_images/macaw.jpg) an image by Roberto Vivancos provided under the [Pexels license](https://www.pexels.com/license/) on [Pexels](https://www.pexels.com/photo/shallow-focus-photography-of-green-yellow-and-blue-bird-2190209).

### Datasets

[CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) are used for the source images via the [standard Torch loaders](https://pytorch.org/vision/main/datasets.html).
