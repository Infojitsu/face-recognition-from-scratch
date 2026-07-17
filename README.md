# Face Recognition from Scratch (Java)

Real-time face detection and recognition where **every ML algorithm is handwritten in pure Java** — no ML libraries, no framework doing the math for me:

- **HOG feature extraction** (Dalal & Triggs, 2005) — gradient computation, orientation histograms with bilinear bin interpolation, L2-Hys block normalization. 8,100-dimensional descriptor per 128x128 window.
- **SVM trained with SMO** (Platt's Sequential Minimal Optimization) — solves the dual problem directly, with linear and sigmoid kernels.
- **Sliding-window detector** — image scale pyramid, double non-maximum suppression (IoU + containment ratio), variance-based fast reject, temporal tracking with coasting.
- **Image processing** — Rec. 601 grayscale conversion, bilinear rescaling, cropping, flip augmentation. All hand-rolled.

OpenCV appears in exactly one place: grabbing frames from the webcam. It is loaded via reflection, so the project **compiles and runs without it** (using a folder of images as the source instead).

## How it works

```
webcam frame
   -> scale pyramid (6 scales)
   -> sliding 128x128 window
   -> variance fast-reject (skips uniform regions: walls, ceiling)
   -> HOG descriptor (8,100 dims)
   -> head / non-head SVM (linear kernel)
   -> non-maximum suppression (IoU + containment)
   -> per-person one-vs-all SVMs (sigmoid kernel)
   -> green box + name drawn on the frame
```

## Engineering notes

The parts of this project I would bring up in an interview:

### A linear-kernel fast path for SMO

Naive SMO caches the full kernel matrix: O(n^2) memory. For n = 3,000 samples of
8,100 dims that is ~72 MB and ~30 minutes of training. For the linear kernel this
is unnecessary — the weight vector `w = sum(alpha_i * y_i * x_i)` can be maintained
incrementally, turning every decision-function evaluation into a single dot product.
Same optimization at inference time: prediction drops from O(N_sv * D) to O(D).
Result: ~200 KB of extra memory and ~2 minutes of training for the same data.

### Why the sigmoid kernel uses alpha = 1/100

The textbook default alpha = 1/d (here 1/8100) pushes `tanh(alpha * <x,y>)` deep
into its linear regime for L2-normalized HOG vectors (typical dot products of
100-150 between similar faces), producing decision scores of ~0.01 and flickering
predictions. With alpha = 1/100 the kernel operates in the nonlinear region of
tanh, and per-person scores separate cleanly.

### Why HOG + SVM replaced skin-color segmentation (v1 -> v2)

The first version of the detector segmented skin-colored pixels and extracted
connected components. It was fast, but fragile: sensitive to skin tone, lighting,
and any skin-colored background object. The HOG + SVM detector is invariant to
all three (gradients, not colors), handles multiple faces in frame, and rejects
background reliably. The price is honest: pure-Java sliding window runs at
roughly 3-5 FPS on a 640x480 stream.

## The app

A Swing GUI with four tabs covering the full workflow:

1. **Capture** — grabs face crops from the webcam and saves them under a
   pseudonym (`faces/<pseudonym>/<pseudonym>_<timestamp>.png`).
2. **View** — browse the collected images per person, delete bad samples or
   remove a person entirely (images, HOG vectors, and classifier).
3. **Train** — three buttons, run in order: train the head detector, extract
   HOG vectors (`hog_vectors/`), train one sigmoid SVM per person
   (one-vs-all, serialized to `classifiers/<pseudonym>.dat`).
4. **Recognize** — live webcam view with detected heads boxed in green and
   labeled with the predicted name.

Trained models are serialized (support vectors only, to keep files small) and
loaded automatically at startup.

## Training data (not included)

This repo ships **no images and no trained models**, by design:

- My own captured faces are personal data.
- The celebrity images I used as additional positives come from a public
  dataset that I am not licensed to redistribute.

The folder layout is created automatically on first run:

```
faces/<pseudonym>/       your captured face crops (Capture tab)
head_images/positive/    images containing heads
head_images/negative/    images without heads (walls, desks, your room)
```

To train it on your own face:

1. Put head images in `head_images/positive/` — your own webcam shots work, or
   a public faces dataset from Kaggle.
   Fill `head_images/negative/` with photos of your room / background.
2. Train tab -> **Train head detector**.
3. Capture tab -> collect a few hundred images per person (at least 2 people).
4. Train tab -> **Extract HOG vectors**, then **Train person classifiers**.
5. Recognize tab -> **Start**.

## Build & run

Requirements: **JDK 8+** (tested on JDK 21). No build system, no IDE — plain `javac`.

```
compile.bat
run.bat
```

Or manually:

```
javac -d bin src\svm\*.java src\svm\core\*.java src\svm\hog\*.java src\svm\svm\*.java src\svm\detector\*.java src\svm\io\*.java src\svm\util\*.java src\svm\ui\*.java
java -Xmx4g --enable-native-access=ALL-UNNAMED -cp "bin;lib\opencv-4120.jar" -Djava.library.path=lib svm.Main
```

`-Xmx4g` matters for large trainings — the sigmoid path caches an n x n kernel matrix.

### OpenCV setup (webcam only)

1. Download OpenCV 4.x for Java from opencv.org.
2. Copy `opencv-4xxx.jar` into `lib/`.
3. Copy the native library into `lib/`: `opencv_java4xxx.dll` (Windows) or
   `libopencv_java4xxx.so` (Linux).

Without OpenCV the app still compiles and runs — use a folder of images as the
frame source instead of the webcam.

## Project structure

```
src/svm/
|-- Main.java
|-- core/       Image (hand-rolled image ops), Rect
|-- hog/        HOG descriptor (Dalal & Triggs)
|-- svm/        SVM + SMO, Kernel interface, Linear & Sigmoid kernels
|-- detector/   HeadDetector (sliding window + NMS), TrainingPipeline
|-- io/         PersonClassifier, Storage (serialization)
|-- util/       ImageSource abstraction: WebcamSource (OpenCV via reflection), FolderSource
|-- ui/         Swing GUI: Capture / View / Train / Recognize tabs
```

## Limitations & ideas

- Sliding window in pure Java is CPU-bound (~3-5 FPS); a stride/scale trade-off
  or integral-image tricks could buy more.
- Hard-negative mining would sharpen the head detector further.
- SVM scores are uncalibrated; Platt scaling would give usable probabilities.
- A HOG visualization mode (gradient rose per cell) is high on my list.

---

Part of my ML-from-scratch portfolio — see also
[nn-digit-recognition](https://github.com/Infojitsu/nn-digit-recognition):
a PyTorch digit recognizer trained entirely on my own hand-drawn dataset.
