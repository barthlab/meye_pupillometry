# [*mEye*](https://www.pupillometry.it): A Deep Learning Tool for Pupillometry

> ⭐ MEYE is available on **MATLAB**! Check it out [here](matlab/README.md)

> Check out [pupillometry.it](https://www.pupillometry.it) for a ready-to-use web-based mEye pupillometry tool!

This branch provides the Python code to make predictions and train/finetune models.
If you are interested in the code of the pupillometry web app, check out the `gh-pages` branch.

## Requirements
You need a Python 3 environment with the following packages installed:

  - tensorflow >= 2.4
  - imageio, imageio-ffmpeg
  - scipy  
  - tqdm

If you want to train models, you also need 

  - adabelief_tf >= 0.2.1
  - pandas
  - sklearn

We provide a [Dockerfile](./Dockerfile) for building an image with docker. It targets `tensorflow/tensorflow:2.13.0-gpu-jupyter`; build and run with GPU access, e.g.:
```bash
docker build -t meye .
docker run --gpus all -it --rm -v "$PWD:/workspace/meye" -w /workspace/meye meye bash
```

## Make Predictions with Pretrained Models

You can make predictions with pretrained models on pre-recorded videos or webcam streams. 

  1. Choose a model. This repo already ships:
       - `models/meye-2025-03-12-Offline.h5` (latest offline-tuned model)
       - `models/meye-2022-01-24.h5` (original release)
     Older models are still available in [Releases](https://github.com/fabiocarrara/meye/releases) (e.g. [`v0.1`](https://github.com/fabiocarrara/meye/tree/v0.1)).
  2. Check out the `pupillometry-offline-videos.ipynb` notebook for a complete example of pupillometry data analysis.
  3. Use the `predict.py` script for a single video/webcam stream. The output CSV defaults to `pupillometry.csv` and the output video is now optional (set `-ov` to write it).

       - ```bash
         # input: webcam (default), CSV only
         predict.py models/meye-2025-03-12-Offline.h5
         ```
     
       - ```bash
         # input: video file with custom ROI and explicit outputs
         predict.py models/meye-2025-03-12-Offline.h5 path/to/video.mp4 -rl 80 -rt 80 -rr 208 -rb 208 -ov video_with_predictions.mp4 -oc pupil_metrics.csv
         ```
       - ```bash
         # check all parameters with
         predict.py -h
         ```

### Batch predictions from metadata

Use `predict_custom_batch.py` to scan a root folder for `EYE_*.npz` metadata files (each containing a `bbox = [x, y, w, h]`) and corresponding `VIDEO_*.mp4` recordings. For every pair it writes `pupil/PUPIL_<SESSION>.csv` (and every few runs a preview `.mp4`).

```bash
python predict_custom_batch.py --root /path/to/sessions --model models/meye-2025-03-12-Offline.h5 --skip-existing
```
Use `--thr` to change the threshold or drop `--skip-existing` to recompute outputs.
    
## Training Models

  1. Download our dataset ([NN_human_mouse_eyes.zip](https://doi.org/10.5281/zenodo.4488164), 246.4 MB) or prepare your dataset following our dataset's structure.
     > If you need to annotate your dataset, check out [pLabeler](https://github.com/LeonardoLupori/pLabeler), a MATLAB software for labeling pupil images.
     
     The dataset should be placed in `data/<dataset_name>`.
     
  2. If you are using a custom dataset, edit `train.py` to perform the train/validation/test split of your data.
     
  3. Train with default parameters:
     ```bash
     python train.py -d data/<dataset_name>
     ```
     
  - For a list of available parameters, run
    ```bash
    python train.py -h
    ```

## MATLAB support
Starting from MATLAB version 2021b, MEYE is also available for use on MATLAB!  
A fully functional class and a tutorial for its use is available [here](matlab/README.md)!


## References

### Dataset
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4488164.svg)](https://doi.org/10.5281/zenodo.4488164)

If you use our dataset, please cite:

     @dataset{raffaele_mazziotti_2021_4488164,
       author       = {Raffaele Mazziotti and Fabio Carrara and Aurelia Viglione and Lupori Leonardo and Lo Verde Luca and Benedetto Alessandro and Ricci Giulia and Sagona Giulia and Amato Giuseppe and Pizzorusso Tommaso},
       title        = {{Human and Mouse Eyes for Pupil Semantic Segmentation}},
       month        = feb,
       year         = 2021,
       publisher    = {Zenodo},
       version      = {1.0},
       doi          = {10.5281/zenodo.4488164},
       url          = {https://doi.org/10.5281/zenodo.4488164}
     }
