# CIL-FS20
## Course Project of the Computational Intelligence Lab FS20 - Road Segmentation on Satellite Images

In this paper, we propose a method to label roads in high-resolution aerial images on a per-pixel basis. With today's
influx of satellite data, manual labeling has become infeasible and we need to rely on Computer Vision for processing.
Our approach uses a U-Net, a fully convolutional neural network which was introduced in 2015 for biomedical image segmentation.
Since neural networks require large amounts of training data, we have devised a method to automatically generate labeled
training data from Google Maps. Such generated data is not as good qualitatively as human-annotated images, but it is
available in almost unlimited amounts.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install -r requirements.txt
```

## Usage
- `baseline2_pipeline.ipynb`: Run full pipeline of baseline 2 model with visuals.
- `final_model.ipynb`: Run full pipeline of final model with visuals.
- `training_final_model.py`: Run only the training of the final model (for cluster).

