# BrainDead 2023: Placement Analytics and Facial Emotion Recognition

Team **Duo Lipa**'s submission to BrainDead 2023. Two independent problems: an exploratory
analysis of campus placement data, and a four-class facial emotion classifier with a GAN used
to generate extra training faces.

Both notebooks were written for Kaggle and read their data from `/kaggle/input/`. Colab badges
are at the top of each notebook.

---

## Problem 1: What actually predicts campus placement

`Problem-1.ipynb`

**Data.** 215 candidate records (`brain-dead-placement-data`): gender, school and degree
percentages, degree stream, work experience, MBA specialisation, employability test score,
placement status, and salary.

**Approach.** Salary nulls are the 67 unplaced candidates, so they are imputed with 0 rather
than dropped or mean-filled, which would have implied unplaced candidates earned something.
Serial number and school board columns are dropped, and higher-secondary percentage is
filtered on the 1.5 IQR rule, the only feature with meaningful outliers. The rest of the
notebook is visual: count plots, distributions, joint plots, a correlation heatmap, and a
pairplot split by placement status.

**What the data says.**

- Academic percentages, especially secondary school and undergraduate, separate placed from
  unplaced candidates more cleanly than anything else.
- Prior work experience does **not** track with placement here. Two thirds of candidates had
  none, and they were placed at least as often.
- Academic performance predicts *whether* someone is placed, but has essentially no
  relationship with *how much* they are paid.
- Placement rates look even across gender, but the highest offers, and the higher average
  offer, went to male candidates.

---

## Problem 2: Facial emotion recognition, and a GAN to feed it

`Problem-2: Classification.ipynb` and `Problem-2: Generate Data.ipynb`

**Data.** `brain-dead-emotion-detection`, four classes (angry, happy, neutral, sad), 369 test
images, loaded at 150x150 through `ImageDataGenerator` with zoom and shift augmentation.

**Model.** Three convolutional blocks (32, 64, 32 filters) each followed by ReLU, max pooling
and batch normalisation, with L2 on the deeper convolutions, then dense layers to a softmax
over four classes. Adam at 1e-3, categorical cross-entropy, `ReduceLROnPlateau` on validation
loss, 20 epochs.

**Results on the test set.**

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| angry | 0.38 | 0.23 | 0.28 | 80 |
| happy | 0.43 | 0.70 | 0.53 | 105 |
| neutral | 0.52 | 0.43 | 0.47 | 102 |
| sad | 0.49 | 0.39 | 0.44 | 82 |
| **accuracy** | | | **0.46** | 369 |
| macro avg | 0.46 | 0.44 | 0.43 | 369 |

> **A note on the 75% figure.** The notebook prints `Test Accuracy: 0.754`, which comes from
> Keras `BinaryAccuracy`. On a four-way softmax that metric scores every output element
> independently, so the three easy zeros in each one-hot row inflate it. The honest number is
> the 0.46 from the classification report above. `CategoricalAccuracy` is the correct metric
> here.

Angry is the weakest class and is most often confused with sad and neutral, which matches the
usual failure mode for small facial-expression datasets: low-arousal negative expressions look
alike at low resolution. The notebook ends by plotting misclassified faces rather than only
reporting aggregate numbers.

**The GAN.** `Problem-2: Generate Data.ipynb` trains a small DCGAN (100-dimensional latent
vector, transposed-convolution generator, convolutional discriminator, RMSprop with clipping
and label smoothing) on the `happy` class at 32x32, to generate additional training faces.
At the resolution and training budget used, samples are shape-and-tone plausible rather than
convincing faces.

---

## Repository layout

```
Problem-1.ipynb                    placement EDA and visual analysis
Problem-2: Classification.ipynb    CNN emotion classifier, metrics, error analysis
Problem-2: Generate Data.ipynb     DCGAN for generating additional face images
Report (Duo Lipa).pdf              competition write-up
LICENSE                            MIT
```

## Running these

The notebooks expect the competition datasets under `/kaggle/input/`. To run elsewhere, change
the two path constants near the top of each notebook:

```python
# Problem 1
placement = pd.read_csv("/kaggle/input/brain-dead-placement-data/Placement_Data_Full_Class.csv")

# Problem 2
train_dataset_path = '/kaggle/input/brain-dead-emotion-detection/brain_dead_emotion_detection/Train'
validation_dataset_path = '/kaggle/input/brain-dead-emotion-detection/brain_dead_emotion_detection/Test'
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `missingno`, `scikit-learn`,
`tensorflow` / `keras`, `opencv-python`, `Pillow`.

## Known limitations

- The classifier is trained from scratch on a small dataset. A fine-tuned pretrained backbone
  would almost certainly beat 0.46, and was out of scope for the competition window.
- Validation and test point at the same directory, so validation metrics during training are
  not held out from the reported test numbers.
- The GAN images were never fed back into the classifier, so the augmentation loop is
  demonstrated but not evaluated.
