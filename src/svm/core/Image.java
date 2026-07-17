package svm.core;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 * Reprezentare proprie a unei imagini RGB + grayscale.
 * Folosita pentru a evita dependenta de algoritmii BufferedImage la prelucrare.
 */
public class Image {

    /** Latime imagine in pixeli */
    private final int width;
    /** Inaltime imagine in pixeli */
    private final int height;
    /** Canal rosu, valori 0..255 */
    private final int[][] r;
    /** Canal verde, valori 0..255 */
    private final int[][] g;
    /** Canal albastru, valori 0..255 */
    private final int[][] b;

    /**
     * Construieste o imagine goala.
     * @param width latime
     * @param height inaltime
     */
    public Image(int width, int height) {
        this.width = width;
        this.height = height;
        this.r = new int[height][width];
        this.g = new int[height][width];
        this.b = new int[height][width];
    }

    /**
     * Converteste un BufferedImage la Image intern.
     * @param bi imaginea Java nativa
     */
    public Image(BufferedImage bi) {
        this.width = bi.getWidth();
        this.height = bi.getHeight();
        this.r = new int[height][width];
        this.g = new int[height][width];
        this.b = new int[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = bi.getRGB(x, y);
                r[y][x] = (rgb >> 16) & 0xFF;
                g[y][x] = (rgb >> 8)  & 0xFF;
                b[y][x] =  rgb        & 0xFF;
            }
        }
    }

    /** @return latimea imaginii */
    public int getWidth()  { return width;  }
    /** @return inaltimea imaginii */
    public int getHeight() { return height; }

    /** @return valoarea rosu la (x,y) */
    public int getR(int x, int y) { return r[y][x]; }
    /** @return valoarea verde la (x,y) */
    public int getG(int x, int y) { return g[y][x]; }
    /** @return valoarea albastru la (x,y) */
    public int getB(int x, int y) { return b[y][x]; }

    /**
     * Seteaza un pixel RGB.
     */
    public void setRGB(int x, int y, int red, int green, int blue) {
        r[y][x] = red;
        g[y][x] = green;
        b[y][x] = blue;
    }

    /**
     * Converteste la matrice de luminanta (grayscale) prin formula Rec. 601:
     * Y = 0.299 R + 0.587 G + 0.114 B
     * @return matrice [height][width] cu valori 0..255
     */
    public double[][] toGrayscale() {
        double[][] gray = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                gray[y][x] = 0.299 * r[y][x] + 0.587 * g[y][x] + 0.114 * b[y][x];
            }
        }
        return gray;
    }

    /**
     * Returneaza un subimagine delimitata de pixelii (x1,y1) - (x2,y2), inclusiv.
     */
    public Image crop(int x1, int y1, int x2, int y2) {
        int w = x2 - x1 + 1;
        int h = y2 - y1 + 1;
        Image sub = new Image(w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int sx = x1 + x, sy = y1 + y;
                sub.r[y][x] = r[sy][sx];
                sub.g[y][x] = g[sy][sx];
                sub.b[y][x] = b[sy][sx];
            }
        }
        return sub;
    }

    /**
     * Scaleaza imaginea la dimensiuni noi folosind interpolare biliniara.
     */
    public Image scale(int newW, int newH) {
        Image out = new Image(newW, newH);
        double fx = (double)(width  - 1) / (newW - 1);
        double fy = (double)(height - 1) / (newH - 1);
        for (int y = 0; y < newH; y++) {
            for (int x = 0; x < newW; x++) {
                double sx = x * fx, sy = y * fy;
                int x0 = (int) Math.floor(sx), x1 = Math.min(x0 + 1, width  - 1);
                int y0 = (int) Math.floor(sy), y1 = Math.min(y0 + 1, height - 1);
                double dx = sx - x0, dy = sy - y0;
                int rr = (int) bilinear(r[y0][x0], r[y0][x1], r[y1][x0], r[y1][x1], dx, dy);
                int gg = (int) bilinear(g[y0][x0], g[y0][x1], g[y1][x0], g[y1][x1], dx, dy);
                int bb = (int) bilinear(b[y0][x0], b[y0][x1], b[y1][x0], b[y1][x1], dx, dy);
                out.setRGB(x, y, rr, gg, bb);
            }
        }
        return out;
    }

    /** Interpolare biliniara intre 4 puncte. */
    private static double bilinear(int q00, int q01, int q10, int q11, double dx, double dy) {
        double a = q00 * (1 - dx) + q01 * dx;
        double b = q10 * (1 - dx) + q11 * dx;
        return a * (1 - dy) + b * dy;
    }

    /**
     * Converteste aceasta imagine intr-un BufferedImage (pentru afisare).
     */
    public BufferedImage toBufferedImage() {
        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = (r[y][x] << 16) | (g[y][x] << 8) | b[y][x];
                bi.setRGB(x, y, rgb);
            }
        }
        return bi;
    }

    /** Incarca imagine dintr-un fisier. */
    public static Image load(File file) throws IOException {
        return new Image(ImageIO.read(file));
    }

    /** Salveaza imaginea ca PNG. */
    public void saveAsPNG(File file) throws IOException {
        ImageIO.write(toBufferedImage(), "png", file);
    }
}
