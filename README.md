# Flooded-Area-Mapping-Form-Satellite-Images-Flooded Area Mapping from Sentinel-1 Satellite Images
Project type: CNN-based image segmentation / water-area mapping  
Dataset: Sen1Floods11 v1.1 — hand-labelled Sentinel-1 SAR imagery  
Notebook: `notebook4a9e86c32d.ipynb` (rename it to `flood\_mapping.ipynb` if you prefer)
1. Project overview
This project compares three approaches for identifying water pixels in Sentinel-1 Synthetic Aperture Radar (SAR) images:
VH threshold baseline — predicts water when the normalized VH backscatter value is below a threshold selected on the validation set.
Compact U-Net — a U-Net segmentation model with a ResNet18 encoder trained from scratch.
Transfer-learning U-Net — a U-Net with an ImageNet-pretrained ResNet18 encoder, fine-tuned for the two-band SAR input.
The task is pixel-level segmentation: the model predicts whether each valid pixel is water or non-water.
> \*\*Important limitation:\*\* The hand-labelled masks identify water/non-water; they do not reliably distinguish newly flooded land from permanent water. Therefore, describe this project as \*water-area mapping\* or \*flood-inundation proxy mapping\*, not as definitive detection of newly flooded areas.
2. Dataset
The project uses the public Sen1Floods11 v1.1 dataset.
Dataset/project page: https://github.com/cloudtostreet/Sen1Floods11
Input: two Sentinel-1 SAR bands, VV and VH (not RGB photographs).
Label values:
`0` = non-water
`1` = water
`-1` = invalid / ignored pixel
The notebook reads the official train, validation, and test split CSV files.
The notebook resizes images to 256 × 256 pixels. SAR values are clipped and normalized. Masks are resized with nearest-neighbour interpolation to preserve label values.
Random flips and 90-degree rotations are used as training-only augmentation.
The notebook downloads the public files when it runs. Internet access to the dataset bucket is required.
3. Environment and requirements
Recommended environment: Google Colab with a T4 GPU.
The notebook installs its main packages in the first code cell. It uses Python with PyTorch, Rasterio, scikit-learn, pandas, Matplotlib, tqdm, segmentation-models-pytorch, and Google Cloud Storage client libraries.
Run in Google Colab
Upload the notebook to Google Colab.
Open Runtime → Change runtime type and select T4 GPU if available.
Choose Runtime → Run all.
Allow the dependency installation, dataset download, training, evaluation, and export cells to finish.
Review the printed metrics and figures. At the end, the notebook creates a ZIP archive of the generated results.
The first run can take a while because the imagery and pretrained weights need to be downloaded and both U-Net models need to be trained. GPU availability and download speed may vary.
Run locally
Use a Python environment with the notebook's dependencies installed, then open the notebook in Jupyter. Update the output/data paths if your environment does not use `/content`. A compatible PyTorch GPU setup is recommended; CPU training may be very slow.
4. How the code runs
Run the notebook from top to bottom. Its main stages are:
Install and import packages — load the libraries needed for image processing, model training, metrics, plots, and downloads.
Configure the run — set the random seed, device, image size (256), batch size (8), epoch count (12), learning rate, and output directory.
Download dataset and official splits — fetch SAR image chips, hand-labelled masks, and split CSV files from the public dataset storage.
Pair images and masks — match each image to its corresponding mask and check that train, validation, and test chip IDs do not overlap.
Preprocess the data — normalize the two SAR bands, resize images and masks, augment training examples, and count valid/non-water/water/ignored pixels.
Visualize an example — show the VV band, VH band, and hand-labelled water mask.
Build and train two U-Net models — train the compact model and the transfer-learning model. Save the weights with the best validation water IoU and save training curves.
Tune the simple baseline — choose the VH threshold using validation data only.
Evaluate on the held-out test set — calculate pixel-level confusion counts, accuracy, precision, recall, Dice/F1, and IoU for all three approaches.
Visualize predictions — compare SAR input, ground truth, and predictions from the threshold baseline and both neural networks.
Export artifacts — save tables, figures, model weights, reports, metadata, and a ZIP archive under the `outputs` directory.
5. Example: running the notebook code
The following is a small example of the kind of output the notebook prints during setup. It is illustrative; the actual device and output path depend on the runtime.
```python
SEED = 42
SIZE = 256
BATCH\_SIZE = 8
EPOCHS = 12
LR = 3e-4

DEVICE = 'cuda' if torch.cuda.is\_available() else 'cpu'
ROOT = Path('/content/flood\_mapping') if Path('/content').exists() else Path('./flood\_mapping')
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist\_ok=True)

print('Device:', DEVICE, '| Output:', OUT)
```
How to read this output:
`Device: cuda` means PyTorch can use a CUDA-compatible GPU. `Device: cpu` means the run is using the CPU.
`Output:` shows the directory where the notebook saves generated files.
`SIZE = 256` means each image and mask is processed at 256 × 256 pixels.
`BATCH\_SIZE = 8` means the model processes up to eight image chips per batch.
`EPOCHS = 12` is the maximum number of training passes over the training set for each model.
`LR = 3e-4` is the learning rate used by AdamW.
Example training log
During training, a line similar in format to this will be printed:
```text
compact\_unet epoch 1/12: train loss ..., val loss ..., val IoU ..., val Dice ...
```
The dots above are placeholders, not measured results. Your notebook will print the actual values.
Train loss: how far the predictions are from the training labels according to the loss function; lower is generally better.
Validation loss: the same loss calculated on the validation split, which is not used for gradient updates.
Validation IoU: overlap between predicted water pixels and labelled water pixels. Higher is better.
Validation Dice: another overlap measure, also called F1 for the binary segmentation task. Higher is better.
If training loss improves but validation loss gets worse, the model may be overfitting. Compare the curves rather than judging the model from one epoch alone.
6. How to understand the evaluation output
The notebook evaluates the chosen models on the held-out test split after training and validation-based selection.
Metrics
Accuracy: proportion of correctly classified valid pixels. It can look high when one class dominates.
Precision: among pixels predicted as water, the proportion that are labelled water. Low precision means more false alarms.
Recall: among labelled water pixels, the proportion correctly found. Low recall means more missed water.
Dice / F1: overlap measure that balances precision and recall.
IoU (Intersection over Union): intersection of predicted and labelled water pixels divided by their union. For the water class, higher IoU indicates better overlap.
TN / FP / FN / TP: true non-water, non-water pixels incorrectly predicted as water, water pixels missed, and correctly predicted water pixels.
The notebook excludes `-1` invalid pixels from the loss and evaluation metrics.
Confusion matrix
Each saved confusion matrix compares actual labels (rows) with predicted labels (columns):
	Predicted non-water	Predicted water
Actual non-water	True negatives (TN)	False positives (FP)
Actual water	False negatives (FN)	True positives (TP)
More values on the diagonal (TN and TP) and fewer off-diagonal errors (FP and FN) generally indicate better classification. Check false positives and false negatives carefully: false positives overstate water extent, while false negatives miss water.
Model comparison table
`model\_comparison.csv` contains the test metrics for `vh\_threshold`, `compact\_unet`, and `transfer\_unet`, sorted by water IoU. Compare IoU and Dice/F1 first, then precision and recall. Do not assume the transfer-learning model is best: use the values from your actual run to decide.
Prediction overlays
The overlay figures show:
the VH SAR input,
the hand-labelled water mask,
the threshold baseline prediction,
the compact U-Net prediction, and
the transfer-learning U-Net prediction.
Red overlay pixels indicate predicted water on valid-label areas. Compare each prediction with the ground truth to see where the models agree, miss water, or label non-water as water.
7. Generated files
After a successful run, the notebook saves artifacts under `flood\_mapping/outputs/` (the notebook's displayed path may differ by environment). Expected files include:
File / pattern	What it contains
`split\_counts.csv`	Number of image chips in train, validation, and test splits
`class\_distribution.csv`	Counts of non-water, water, and ignored training pixels
`class\_distribution.png`	Plot of the training label distribution
`sample\_dataset.png`	Example VV band, VH band, and hand-labelled mask
`compact\_unet\_history.csv`	Per-epoch training and validation metrics for the compact U-Net
`transfer\_unet\_history.csv`	Per-epoch training and validation metrics for the transfer-learning U-Net
`compact\_unet\_curves.png`	Loss, accuracy, and IoU curves for the compact U-Net
`transfer\_unet\_curves.png`	Loss, accuracy, and IoU curves for the transfer-learning U-Net
`baseline\_threshold\_tuning.png`	Validation IoU across candidate VH thresholds
`model\_comparison.csv`	Test metrics for all three approaches
`\*\_confusion\_matrix.png`	Pixel-level confusion matrix for each approach
`\*\_classification\_report.txt`	Pixel-level classification report for each approach
`test\_overlay\_\*.png`	Test examples comparing ground truth and predictions
`\*\_best.pth`	Best-validation-IoU weights for each U-Net
`run\_metadata.json`	Configuration, selected threshold, device, and split counts
`flood\_mapping\_results.zip`	ZIP archive of the contents of the outputs directory
The notebook also displays a download link for the results ZIP at the end. File creation depends on the corresponding cells completing successfully.
8. How to write up your results
After you run the notebook, replace the placeholders in a results summary with your actual values. Do not report example or invented numbers.
Suggested format:
> On the Sen1Floods11 v1.1 hand-labelled test split, the VH threshold baseline achieved a water IoU of \*\*\[insert value]\*\* and Dice/F1 of \*\*\[insert value]\*\*. The compact U-Net achieved a water IoU of \*\*\[insert value]\*\*, while the transfer-learning U-Net achieved \*\*\[insert value]\*\*. The approach with the highest test water IoU had the strongest pixel-level overlap with the hand-labelled water masks in this run. Precision and recall were also examined to understand false alarms and missed water pixels.
Use the confusion matrices and overlays to explain the kinds of errors you observed. Discuss limitations, including permanent water being included in the water class and possible spatial similarity between nearby chips.
9. Reproducibility and limitations
The random seed is set to 42, but exact results can still vary across hardware and software versions.
The baseline threshold is selected on validation data; it should not be tuned on the test split.
The test split is intended for final evaluation, not repeated hyperparameter tuning.
Invalid mask pixels (`-1`) are excluded from training loss and metrics.
Accuracy is secondary to water IoU and Dice when the class distribution is imbalanced.
ImageNet-pretrained weights are used for the transfer-learning encoder; the network is adapted to accept two SAR input bands.
Dataset access, package versions, GPU availability, and runtime limits can affect whether the notebook completes successfully.
10. Troubleshooting
Notebook stays queued for GPU: wait briefly, then retry later or switch to CPU if available (training may be much slower).
Dataset download fails: check that the runtime has internet access and rerun the download cell.
Out-of-memory error: reduce `BATCH\_SIZE` (for example, from 8 to 4) and rerun from the appropriate setup/training cells.
Missing package error: rerun the first installation cell, then restart the runtime if necessary.
No output figures or CSVs: ensure the relevant cells ran to completion and check the printed `OUT` directory.
Kernel/runtime reset: rerun the notebook from the beginning so data, models, and variables are recreated.
11. References
Cloud to Street, Sen1Floods11 dataset and project repository: https://github.com/cloudtostreet/Sen1Floods11
PyTorch documentation: https://pytorch.org/docs/
Segmentation Models PyTorch documentation: https://smp.readthedocs.io/
