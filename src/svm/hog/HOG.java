package svm.hog;

import svm.core.Image;

/**
 * Histogram of Oriented Gradients (HOG) extractor, implemented entirely from scratch,
 * following the original Dalal &amp; Triggs (2005) algorithm.
 *
 * Default parameters for a 128x128 image:
 *  - cell size 8x8 pixels
 *  - block size 2x2 cells (16x16 pixels)
 *  - block stride 1 cell (overlapping windows)
 *  - 9 unsigned orientations (0..180 degrees)
 *
 * Final vector for 128x128: 15*15 blocks * 36 components = 8100 values.
 */
public class HOG {

    /** Cell size (pixels) */
    public static final int CELL = 8;
    /** Block size in cells */
    public static final int BLOCK = 2;
    /** Number of orientations (0..180 degrees) */
    public static final int BINS = 9;

    /** Expected image 128x128 */
    public static final int IMG = 128;

    /**
     * Computes the HOG vector for an image.
     * @param img image (will be converted to grayscale)
     * @return normalized feature vector
     */
    public static double[] compute(Image img) {
        double[][] gray = img.toGrayscale();
        return compute(gray);
    }

    /**
     * Computes the HOG vector on an arbitrary grayscale matrix.
     */
    public static double[] compute(double[][] gray) {
        int H = gray.length;
        int W = gray[0].length;

        // 1. Gradients Gx and Gy with masks [-1 0 1] and [-1 0 1]^T
        double[][] gx = new double[H][W];
        double[][] gy = new double[H][W];
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int xm = Math.max(x - 1, 0);
                int xp = Math.min(x + 1, W - 1);
                int ym = Math.max(y - 1, 0);
                int yp = Math.min(y + 1, H - 1);
                gx[y][x] = gray[y][xp] - gray[y][xm];
                gy[y][x] = gray[yp][x] - gray[ym][x];
            }
        }

        // 2. Magnitude and orientation (0..180 degrees, unsigned)
        double[][] mag = new double[H][W];
        double[][] ang = new double[H][W];
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                mag[y][x] = Math.sqrt(gx[y][x] * gx[y][x] + gy[y][x] * gy[y][x]);
                double a = Math.toDegrees(Math.atan2(gy[y][x], gx[y][x]));
                if (a < 0) a += 180.0;     // unsigned
                if (a >= 180.0) a -= 180.0;
                ang[y][x] = a;
            }
        }

        // 3. Per-cell histograms (H/CELL x W/CELL cells, BINS values each)
        int cellsY = H / CELL;
        int cellsX = W / CELL;
        double[][][] hist = new double[cellsY][cellsX][BINS];
        double binSize = 180.0 / BINS;

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int cy = y / CELL;
                int cx = x / CELL;
                if (cy >= cellsY || cx >= cellsX) continue;
                // Linear interpolation between two adjacent bins
                double a = ang[y][x];
                double m = mag[y][x];
                double bf = a / binSize;         // fractional position in bins
                int b0 = (int) Math.floor(bf - 0.5);
                int b1 = b0 + 1;
                double f = (bf - 0.5) - b0;      // relative distance from b0
                int i0 = ((b0 % BINS) + BINS) % BINS;
                int i1 = ((b1 % BINS) + BINS) % BINS;
                hist[cy][cx][i0] += m * (1.0 - f);
                hist[cy][cx][i1] += m * f;
            }
        }

        // 4. Normalization over 2x2-cell blocks with the L2-Hys scheme
        int blocksY = cellsY - BLOCK + 1;
        int blocksX = cellsX - BLOCK + 1;
        int blockLen = BLOCK * BLOCK * BINS;
        double[] feat = new double[blocksY * blocksX * blockLen];
        int idx = 0;
        double eps = 1e-6;
        double clip = 0.2;

        for (int by = 0; by < blocksY; by++) {
            for (int bx = 0; bx < blocksX; bx++) {
                double[] v = new double[blockLen];
                int k = 0;
                for (int ey = 0; ey < BLOCK; ey++) {
                    for (int ex = 0; ex < BLOCK; ex++) {
                        for (int bi = 0; bi < BINS; bi++) {
                            v[k++] = hist[by + ey][bx + ex][bi];
                        }
                    }
                }
                // L2 norm
                double norm = 0;
                for (double vi : v) norm += vi * vi;
                norm = Math.sqrt(norm + eps);
                for (int i = 0; i < v.length; i++) v[i] /= norm;
                // clip (Hys)
                for (int i = 0; i < v.length; i++) if (v[i] > clip) v[i] = clip;
                // renormalize
                norm = 0;
                for (double vi : v) norm += vi * vi;
                norm = Math.sqrt(norm + eps);
                for (int i = 0; i < v.length; i++) v[i] /= norm;
                // copy into the final vector
                for (double vi : v) feat[idx++] = vi;
            }
        }

        return feat;
    }
}
