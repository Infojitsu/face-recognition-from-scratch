package svm.hog;

import svm.core.Image;

/**
 * Extractor Histogram of Oriented Gradients (HOG), implementat complet de la zero,
 * dupa algoritmul original Dalal &amp; Triggs (2005).
 *
 * Parametri impliciti pentru imagine 128x128:
 *  - dimensiune celula 8x8 pixeli
 *  - dimensiune bloc 2x2 celule (16x16 pixeli)
 *  - stride bloc 1 celula (ferestre suprapuse)
 *  - 9 orientari unsigned (0..180 grade)
 *
 * Vectorul final pentru 128x128: 15*15 blocuri * 36 componente = 8100 valori.
 */
public class HOG {

    /** Dimensiune celula (pixeli) */
    public static final int CELL = 8;
    /** Dimensiune bloc in celule */
    public static final int BLOCK = 2;
    /** Numar de orientari (0..180 grade) */
    public static final int BINS = 9;

    /** Imagine asteptata 128x128 */
    public static final int IMG = 128;

    /**
     * Calculeaza vectorul HOG pentru o imagine.
     * @param img imagine (va fi convertita in grayscale)
     * @return vector de trasaturi normalizat
     */
    public static double[] compute(Image img) {
        double[][] gray = img.toGrayscale();
        return compute(gray);
    }

    /**
     * Calculeaza vectorul HOG pe o matrice grayscale oarecare.
     */
    public static double[] compute(double[][] gray) {
        int H = gray.length;
        int W = gray[0].length;

        // 1. Gradient Gx si Gy cu masti [-1 0 1] si [-1 0 1]^T
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

        // 2. Magnitudine si orientare (0..180 grade, unsigned)
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

        // 3. Histograme per celula (H/CELL x W/CELL celule, BINS valori fiecare)
        int cellsY = H / CELL;
        int cellsX = W / CELL;
        double[][][] hist = new double[cellsY][cellsX][BINS];
        double binSize = 180.0 / BINS;

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int cy = y / CELL;
                int cx = x / CELL;
                if (cy >= cellsY || cx >= cellsX) continue;
                // Interpolare liniara intre doua bin-uri adiacente
                double a = ang[y][x];
                double m = mag[y][x];
                double bf = a / binSize;         // pozitie fractionara in bin-uri
                int b0 = (int) Math.floor(bf - 0.5);
                int b1 = b0 + 1;
                double f = (bf - 0.5) - b0;      // distanta relativa de la b0
                int i0 = ((b0 % BINS) + BINS) % BINS;
                int i1 = ((b1 % BINS) + BINS) % BINS;
                hist[cy][cx][i0] += m * (1.0 - f);
                hist[cy][cx][i1] += m * f;
            }
        }

        // 4. Normalizare pe blocuri 2x2 celule cu schema L2-Hys
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
                // renormalizeaza
                norm = 0;
                for (double vi : v) norm += vi * vi;
                norm = Math.sqrt(norm + eps);
                for (int i = 0; i < v.length; i++) v[i] /= norm;
                // copiaza in vector final
                for (double vi : v) feat[idx++] = vi;
            }
        }

        return feat;
    }
}
