package svm.detector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import svm.core.Image;
import svm.core.Rect;
import svm.hog.HOG;
import svm.svm.SVM;

/**
 * Head detector based on sliding window + HOG + SVM.
 *
 * The strategy (Dalal &amp; Triggs 2005, the classic):
 *   1. Build a pyramid of image scales (so we can detect faces
 *      of different sizes with the same 128x128 classifier).
 *   2. At each scale, scan the image with a 128x128 window and
 *      a stride of STRIDE pixels.
 *   3. For each position: extract the HOG, apply the SVM trained
 *      on "head vs non-head". If the score is positive, it's a candidate.
 *   4. Non-Maximum Suppression: remove overlapping detections, keep
 *      the one with the higher SVM score.
 *
 * Advantages over skin-color:
 *   - Invariant to skin tone and lighting (HOG uses gradients)
 *   - Invariant to background - curtains, walls etc. no longer matter
 *   - Can detect multiple faces at once
 *
 * Drawback: slower. In pure Java, about 3-5 FPS for a 640x480 frame.
 */
public class HeadDetector {

    /** Pyramid scales - cover from a distant head (153px) up to a head
     *  close to the camera (640px) on an HD webcam. Scales above min(W,H) are
     *  ignored in detect() anyway, so there is no cost for a VGA webcam. */
    private static final double[] SCALES = { 1.2, 1.6, 2.2, 3.0, 4.0, 5.0 };
    /** Window stride as a fraction of the window size */
    private static final double STRIDE_RATIO = 0.25;
    /** SVM decision threshold - stricter, rejects marginal detections (neck, partial forehead etc). */
    private double threshold = 0.25;
    /** Maximum overlap for NMS (IoU) - smaller = suppresses overlapping squares more aggressively */
    private static final double NMS_OVERLAP = 0.15;
    /** Intersection-over-min-area threshold - suppresses when a small square is
     *  largely contained inside another (even if not perfectly). 0.35 suppresses
     *  pretty much any square nested inside a larger one. */
    private static final double NMS_CONTAIN = 0.35;
    /** Minimum side length of the detected square */
    private static final int MIN_SIDE = 100;
    /** Minimum grayscale pixel variance inside the window (0-255).
     *  Below this threshold the region is too uniform (wall, wardrobe, ceiling) -
     *  it cannot possibly be a head. A face with brow+skin+eyes generates >1000. */
    private static final double MIN_PIXEL_VARIANCE = 400.0;

    /** The head/non-head SVM classifier - REQUIRED for detection. */
    private SVM classifier;

    /** Temporal stabilization for the webcam stream */
    private Rect lastHead;
    private int framesSinceLastSeen;
    private static final int MAX_COAST_FRAMES = 3;
    private boolean trackingEnabled = false;

    /** SVM scores for the last detect() call, in the order of the returned detections. */
    private List<Double> lastDetectScores = new ArrayList<>();

    public HeadDetector() {}

    public HeadDetector(SVM classifier) {
        this.classifier = classifier;
    }

    /** Sets the classifier (after training or loading). */
    public void setVerifier(SVM c) { this.classifier = c; }

    /** Sets the SVM threshold (default 0). Smaller = more detections. */
    public void setThreshold(double t) { this.threshold = t; }

    /** @return the current SVM decision threshold. */
    public double getThreshold() { return threshold; }

    /** @return the list of SVM scores for the last detect(), in the same
     *  order as the returned Rects. The list is empty if detect() returned
     *  nothing. */
    public List<Double> getLastDetectScores() { return lastDetectScores; }

    public void setTrackingEnabled(boolean on) {
        this.trackingEnabled = on;
        if (!on) { lastHead = null; framesSinceLastSeen = 0; }
    }

    public void resetTracking() {
        this.lastHead = null;
        this.framesSinceLastSeen = 0;
    }

    /**
     * Inner class for a detection (with SVM score).
     */
    private static class Detection {
        Rect rect;
        double score;
        Detection(Rect r, double s) { rect = r; score = s; }
    }

    /**
     * Detects all heads in the image.
     * @return list of Rects (squares), sorted by SVM score descending
     */
    public List<Rect> detect(Image img) {
        if (classifier == null) {
            // Without a trained SVM we cannot detect.
            return Collections.emptyList();
        }

        List<Detection> all = new ArrayList<>();
        int W = img.getWidth();
        int H = img.getHeight();

        // Pyramid traversal
        for (double scale : SCALES) {
            int winSize = (int)(HOG.IMG * scale);
            if (winSize > Math.min(W, H)) break;
            if (winSize < MIN_SIDE) continue;

            int step = Math.max(8, (int)(winSize * STRIDE_RATIO));

            for (int y = 0; y + winSize < H; y += step) {
                for (int x = 0; x + winSize < W; x += step) {
                    Image win = img.crop(x, y, x + winSize - 1, y + winSize - 1)
                                   .scale(HOG.IMG, HOG.IMG);
                    double[][] gray = win.toGrayscale();
                    // Fast filter: nearly uniform regions (walls, wardrobe,
                    // ceiling) have low variance => skip HOG and SVM.
                    if (pixelVariance(gray) < MIN_PIXEL_VARIANCE) continue;
                    double[] feat = HOG.compute(gray);
                    double sc = classifier.decisionFunction(feat);
                    if (sc > threshold) {
                        all.add(new Detection(new Rect(x, y, winSize, winSize), sc));
                    }
                }
            }
        }

        // Non-Maximum Suppression
        List<Detection> kept = nms(all);

        // Sort by score descending
        kept.sort(new Comparator<Detection>() {
            @Override public int compare(Detection a, Detection b) {
                return Double.compare(b.score, a.score);
            }
        });

        List<Rect> out = new ArrayList<>();
        lastDetectScores.clear();
        for (Detection d : kept) {
            out.add(d.rect);
            lastDetectScores.add(d.score);
        }
        return out;
    }

    /**
     * Non-Maximum Suppression: if two detections overlap heavily,
     * keep the one with the higher score.
     */
    private List<Detection> nms(List<Detection> dets) {
        List<Detection> sorted = new ArrayList<>(dets);
        sorted.sort(new Comparator<Detection>() {
            @Override public int compare(Detection a, Detection b) {
                return Double.compare(b.score, a.score);
            }
        });
        List<Detection> kept = new ArrayList<>();
        boolean[] suppressed = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (suppressed[i]) continue;
            Detection a = sorted.get(i);
            kept.add(a);
            for (int j = i + 1; j < sorted.size(); j++) {
                if (suppressed[j]) continue;
                Detection b = sorted.get(j);
                // Suppress if: (1) high IoU OR (2) one is almost
                // contained in the other (useful for different scales where IoU
                // is small but one is inside the other).
                if (iou(a.rect, b.rect) > NMS_OVERLAP
                        || containmentRatio(a.rect, b.rect) > NMS_CONTAIN) {
                    suppressed[j] = true;
                }
            }
        }
        return kept;
    }

    /** Grayscale pixel variance - an indicator of texture/uniformity. */
    private static double pixelVariance(double[][] gray) {
        int h = gray.length;
        int w = gray[0].length;
        double sum = 0, sum2 = 0;
        int n = h * w;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double v = gray[y][x];
                sum  += v;
                sum2 += v * v;
            }
        }
        double mean = sum / n;
        return sum2 / n - mean * mean;
    }

    /** Intersection over Union (IoU) for two rectangles. */
    private static double iou(Rect a, Rect b) {
        int ix1 = Math.max(a.x, b.x);
        int iy1 = Math.max(a.y, b.y);
        int ix2 = Math.min(a.x + a.w, b.x + b.w);
        int iy2 = Math.min(a.y + a.h, b.y + b.h);
        int iw = Math.max(0, ix2 - ix1);
        int ih = Math.max(0, iy2 - iy1);
        int inter = iw * ih;
        int union = a.w * a.h + b.w * b.h - inter;
        return union == 0 ? 0 : (double) inter / union;
    }

    /**
     * The ratio intersection / min(area_a, area_b).
     * If a square is completely contained in another, this returns 1.0.
     * Useful when IoU-based NMS misses squares of very different scales.
     */
    private static double containmentRatio(Rect a, Rect b) {
        int ix1 = Math.max(a.x, b.x);
        int iy1 = Math.max(a.y, b.y);
        int ix2 = Math.min(a.x + a.w, b.x + b.w);
        int iy2 = Math.min(a.y + a.h, b.y + b.h);
        int iw = Math.max(0, ix2 - ix1);
        int ih = Math.max(0, iy2 - iy1);
        int inter = iw * ih;
        int minArea = Math.min(a.w * a.h, b.w * b.h);
        return minArea == 0 ? 0 : (double) inter / minArea;
    }

    /**
     * Requirement (2): the highest-scoring head, scaled to 128x128.
     * NEW: returns a crop directly from the image (already square, no
     * extra scaling needed if it is already 128x128).
     */
    public Image largestHead128(Image img) {
        Rect best = largestHeadRect(img);
        if (best == null) return null;
        return img.crop(best.x, best.y, best.x + best.w - 1, best.y + best.h - 1)
                  .scale(HOG.IMG, HOG.IMG);
    }

    /**
     * Returns the highest-scoring square. With tracking enabled: coasts for
     * a few frames when the detector momentarily misses.
     */
    public Rect largestHeadRect(Image img) {
        return selectPrimary(detect(img));
    }

    /**
     * Variant for when the caller already has the detect() result
     * - avoids running the sliding-window pipeline twice.
     * Also updates the tracking (coasting) state.
     */
    public Rect selectPrimary(List<Rect> heads) {
        if (heads.isEmpty()) {
            if (trackingEnabled && lastHead != null
                    && framesSinceLastSeen < MAX_COAST_FRAMES) {
                framesSinceLastSeen++;
                return lastHead;
            }
            if (framesSinceLastSeen >= MAX_COAST_FRAMES) lastHead = null;
            return null;
        }
        Rect best = heads.get(0);
        framesSinceLastSeen = 0;
        lastHead = best;
        return best;
    }
}
