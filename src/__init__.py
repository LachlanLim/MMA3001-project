"""
# Pork-Rasher Packaging Defect Detection

This project teaches a computer to spot faulty packs of frozen pork
rashers from a photo. Given an image of a pack, the trained model draws
a box around each defect it finds and labels it with the defect type
and how confident it is.

| | |
|---|---|
| **Input** | A colour photo of one pork-rasher pack (JPEG or PNG, any size) |
| **Output** | A box around each suspected defect, with its type and a confidence score from 0 to 1 |
| **Defect types** | `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle` |

## How it would be used

On a packing line, a camera photographs each pack as it passes. The
model checks the photo in about 10 milliseconds. If it finds a defect
above a chosen confidence level, the pack is flagged for a person to
check or is pushed off the line. Packs with no detections continue to
shipping.

The model is meant to **help** an inspector, not replace one: it finds
about 9 in 10 unsealed packs on unseen images, but struggles with rare
defect types it saw few examples of (see `src.evaluate`).

## Using the trained model

Once training has finished, checking a new photo takes a few lines:

```python
from ultralytics import YOLO

model = YOLO("models/yolo11n_640.pt")
result = model.predict("pack.jpg", conf=0.25)[0]

for box in result.boxes:
    defect = model.names[int(box.cls)]
    confidence = float(box.conf)
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(defect, round(confidence, 2), [round(x1), round(y1), round(x2), round(y2)])
```

Real output for one pack from the test set (a 720 × 540 photo whose
label marks one packaging error):

```text
packaging-error 0.41 [0, 370, 321, 481]
packaging-error 0.28 [2, 382, 176, 468]
```

The model found the packaging error near the bottom-left of the
photo. The second, weaker box covers part of the same defect. Raising
`conf` to 0.3 would remove it, at the risk of missing faint defects.
Choosing this threshold is a trade-off between false alarms and missed
defects.

## Method

### How it works

The model is **YOLO11** ("You Only Look Once", version 11), a
convolutional neural network for object detection. It looks at the
whole photo in a single pass, in three stages:

1. **Backbone:** layers of small filters pick out visual features:
   edges and textures first, then shapes such as a film edge or a strip
   of meat.
2. **Neck:** combines features at several scales, so both small defects
   (loose meat) and large ones (an unsealed edge) can be found.
3. **Head:** at every position in the photo, predicts whether a defect
   is there, which type it is, and where its box edges are.

This produces many overlapping candidate boxes. **Non-maximum
suppression** then keeps only the most confident box wherever boxes of
the same type overlap by more than 70% (IoU above 0.7).

**Training is a numerical optimisation problem.** The model has 2.6
million adjustable numbers (weights). A **loss function** scores how
wrong its predictions are on the training photos:

```text
loss = 7.5 × box error  +  0.5 × class error  +  1.5 × box-edge error
       (CIoU)              (cross-entropy)        (distribution focal loss)
```

Training repeatedly lowers this loss by **gradient descent**:

1. **Backpropagation** works out how much each weight affects the loss
   (the gradient).
2. Each weight is nudged a small step in the direction that reduces it.

Ultralytics' automatic setting chose the **AdamW** optimiser, which
adapts the step size for each weight separately. Its settings were:

- a learning rate of 0.001111, warmed up over the first 3 epochs and then
  decreased linearly to 1% of that value;
- weight decay of 0.0005, which discourages extreme weights and reduces
  over-fitting;
- gradients from 8 batches of 8 photos added together before each
  update. This behaves like a batch of 64, which gives steadier steps
  than 8 photos alone.

**Numerical stability:**

- **Mixed precision:** training uses 16-bit numbers where possible for
  speed. Ultralytics scales the loss up before backpropagation so that
  very small gradients are not rounded to zero (**loss scaling**).
- **Warm-up:** starting with tiny steps avoids large, unstable updates
  while the new detection head is still random.
- **Fixed seed (42):** the random choices made during training repeat
  from run to run.

### Why this method

- **Object detection, not whole-photo classification:** a pack can have
  several defects, and showing *where* each one is lets an inspector
  confirm it in a glance.
- **Single pass, so it is fast:** around 10 ms per photo (see
  `src.evaluate`), quick enough for a moving line.
  Two-stage detectors such as Faster R-CNN first propose regions and
  then classify each one, which is slower.
- **Transfer learning:** starting from weights pretrained on COCO, a
  large public photo collection, means 3349 photos is enough. Training
  from scratch would need far more.
- **Mature, reproducible tooling:** the Ultralytics library handles
  training, evaluation and export with fixed, recorded settings.

### Assumptions

- The labels are correct and consistent. This is checked by
  `src.validate_annotations` and by eye with `src.visualise_labels`.
- New packs will look like the dataset photos: same camera angle,
  lighting and packaging.
- Each defect is visible in a single colour photo of the pack.
- The training, validation and test photos come from the same source.
  This keeps the comparison fair, but it cannot show how the model
  copes with a different production line.

### Settings

| Setting | Value | Reason |
|---|---|---|
| Model | YOLO11n (nano), pretrained on COCO | Most accurate of the four tested; smallest |
| Resolution | 640 px | 480 px lost accuracy; 800 px gained none |
| Epochs | 50 | Validation accuracy peaked between epochs 22 and 36 |
| Batch size | 8 (accumulated to 64) | Fits in GPU memory at every resolution |
| Optimiser | AdamW, learning rate 0.001111 | Chosen automatically by Ultralytics for short training runs |
| Augmentation | Mosaic, colour jitter, flips, shifts, scaling | Teaches the model to ignore position and lighting changes |
| Confidence threshold | 0.15 | Highest threshold that rejects at least 95% of defective packs on the validation photos (see `src.operating_point`) |

Augmentation randomly alters each training photo every epoch. **Mosaic**
stitches four photos into one; **colour jitter** varies hue, saturation
and brightness; flips, shifts and scaling move the defects around. The
model therefore sees endless variations instead of the same 3349
photos. Mosaic is switched off for the last 10 epochs, so the model
finishes training on realistic single photos.

### Implementation

Each step is a short Python script built on the Ultralytics library,
which runs on PyTorch. The scripts are listed under *Modules* below. Each
experiment script changes only one setting from the baseline, and
`src.evaluate` regenerates every reported number from the saved models.

### Limitations

- **Rare defects are poorly learned.** Twisted-meat (72 training
  examples) was almost never found on the test photos.
- **Wrinkle is untested.** It has no examples in the validation or test
  photos.
- **Each experiment was run once.** Small differences, such as 0.647 vs
  0.646 mAP50, are within run-to-run noise.
- **No warning for unusual photos.** The model will still make confident
  predictions on unfamiliar photos (another product, a different
  camera), and cannot flag them as unusual.
- **The pass/fail target was only nearly met.** The threshold was chosen
  to reject at least 95% of defective packs on the validation photos. On
  the unseen test photos it rejected 92.6% (63 of 68), and wrongly
  rejected 3 of the 12 good packs. A real plant would set its own target.
- **The test score is uncertain.** With only 80 test photos, the 95%
  confidence interval for mAP50 runs from 0.41 to 0.67.

## Workflow

Each module is a script, run from the project root with
`python src/<module>.py`, in this order:

```mermaid
flowchart LR
    A[download_data] --> B[inspect_dataset]
    B --> C[validate_annotations]
    C --> D[visualise_labels]
    D --> E[train_yolo11n<br/>train_yolo11n_480<br/>train_yolo11n_800<br/>train_yolo11s]
    E --> F[evaluate]
    F --> G[operating_point]
```

## Modules

**1. Data preparation:** get the photos and check the labels are correct

- `src.download_data`: download the dataset from Roboflow
- `src.inspect_dataset`: show the defect types and count the photos
- `src.validate_annotations`: check every label for mistakes
- `src.visualise_labels`: draw the labelled boxes on photos to check
  them by eye

**2. Training experiments:** teach the model, testing which settings work best

- `src.train_yolo11n`: the baseline (small model, 640-pixel photos)
- `src.train_yolo11n_480`: does a lower resolution lose accuracy?
- `src.train_yolo11n_800`: does a higher resolution gain accuracy?
- `src.train_yolo11s`: does a larger model do better?
- `src.resume_yolo11s`: continue a training run that was interrupted

**3. Evaluation and reporting:** measure how well it works

- `src.evaluate`: compare accuracy, speed and memory use, then give the
  chosen model its final test
- `src.operating_point`: choose the confidence threshold for use on a
  line, put error bars on the test score, and test whether batching
  photos removes the speed limit
- `src.make_test_report`: turn the automated test results into a
  readable web page

## Glossary

- **Bounding box**: The rectangle drawn around a defect, given by its
  corner positions (x1, y1, x2, y2) in pixels.
- **YOLO label format**: How the true boxes are stored: one line per
  defect, `class_id x_center y_center width height`. The four numbers
  are fractions of the image size (0 to 1), so `0.5` means halfway
  across.
- **Class**: A defect type. Each has a number: 0 `loose-meat`, 1
  `packaging-error`, 2 `twisted-meat`, 3 `unsealed`, 4 `wrinkle`.
- **Confidence**: How sure the model is about a detection, from 0 to 1.
  Detections below a chosen threshold (`conf`) are thrown away.
- **Training / validation / test split**: The photos are divided three
  ways. The model *learns* from the training photos (3349), settings are
  *chosen* using the validation photos (120), and the final model is
  *checked once* on the test photos (80) it has never been used with,
  like a final exam.
- **Epoch**: One full pass through all the training photos. Each model
  here was trained for 50 epochs.
- **Batch size**: How many photos the model looks at together before
  updating itself (8).
- **Pretrained weights**: A model that has already learned general
  shapes and textures from a large public photo collection (COCO).
  Starting from these means far fewer pork-pack photos are needed than
  training from scratch.
- **Seed**: A fixed starting number for the random choices made during
  training (42), so a run can be repeated.
- **Resolution (imgsz)**: The size, in pixels, that each photo is
  resized to before the model sees it. Bigger shows more detail but
  needs more computation.
- **IoU (intersection over union)**: How much a predicted box overlaps
  the true box, from 0 (no overlap) to 1 (perfect match).
- **Precision**: Of the defects the model reported, the fraction that
  were real. Low precision means many false alarms.
- **Recall**: Of the real defects, the fraction the model found. Low
  recall means missed defects.
- **mAP50**: The main accuracy score, from 0 to 1. It combines precision
  and recall across all confidence levels, averaged over the defect
  types, counting a detection as correct if its IoU with the true box is
  at least 0.5.
- **mAP50-95**: A stricter version of mAP50 that also rewards boxes
  drawn tightly around the defect (IoU thresholds from 0.5 up to 0.95).
- **Loss function**: A single number measuring how wrong the model's
  predictions are. Training works by making it smaller.
- **Gradient descent**: Repeatedly nudging every weight in the
  direction that lowers the loss. The **learning rate** sets the step
  size.
- **Over-fitting**: When a model memorises its training photos instead
  of learning general rules, so it does worse on new photos.
- **Augmentation**: Randomly altering training photos (flips, colour
  changes and so on) so the model learns from more variety.
- **Inference**: Using the trained model to check a photo, as opposed to
  training it.
- **GFLOPs**: Billions of arithmetic operations the model needs to check
  one photo, a hardware-independent measure of how much work it does.
- **Checkpoint (`best.pt`, `last.pt`)**: Saved copies of the model.
  `best.pt` is from the epoch that scored highest on the validation
  photos; `last.pt` is from the final epoch.

See the project README for setup instructions.
"""
