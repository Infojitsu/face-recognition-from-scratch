package svm.core;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 * Custom representation of an RGB + grayscale image.
 * Used to avoid depending on BufferedImage algorithms during processing.
 */
public class Image {

    /** Image width in pixels */
    private final int width;
    /** Image height in pixels */
    private final int height;
    /** Red channel, values 0..255 */
    private final int[][] r;
    /** Green channel, values 0..255 */
    private final int[][] g;
    /** Blue channel, values 0..255 */
    private final int[][] b;

    /**
     * Builds an empty image.
     * @param width width
     * @param height height
     */
    public Image(int width, int height) {
        this.width = width;
        this.height = height;
        this.r = new int[height][width];
        this.g = new int[height][width];
        this.b = new int[height][width];
    }

    /**
     * Converts a BufferedImage to the internal Image.
     * @param bi the native Java image
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

    /** @return the image width */
    public int getWidth()  { return width;  }
    /** @return the image height */
    public int getHeight() { return height; }

    /** @return the red value at (x,y) */
    public int getR(int x, int y) { return r[y][x]; }
    /** @return the green value at (x,y) */
    public int getG(int x, int y) { return g[y][x]; }
    /** @return the blue value at (x,y) */
    public int getB(int x, int y) { return b[y][x]; }

    /**
     * Sets an RGB pixel.
     */
    public void setRGB(int x, int y, int red, int green, int blue) {
        r[y][x] = red;
        g[y][x] = green;
        b[y][x] = blue;
    }

    /**
     * Converts to a luminance (grayscale) matrix using the Rec. 601 formula:
     * Y = 0.299 R + 0.587 G + 0.114 B
     * @return [height][width] matrix with values 0..255
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
     * Returns a sub-image bounded by pixels (x1,y1) - (x2,y2), inclusive.
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
     * Scales the image to new dimensions using bilinear interpolation.
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

    /** Bilinear interpolation between 4 points. */
    private static double bilinear(int q00, int q01, int q10, int q11, double dx, double dy) {
        double a = q00 * (1 - dx) + q01 * dx;
        double b = q10 * (1 - dx) + q11 * dx;
        return a * (1 - dy) + b * dy;
    }

    /**
     * Converts this image to a BufferedImage (for display).
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

    /** Loads an image from a file. */
    public static Image load(File file) throws IOException {
        return new Image(ImageIO.read(file));
    }

    /** Saves the image as PNG. */
    public void saveAsPNG(File file) throws IOException {
        ImageIO.write(toBufferedImage(), "png", file);
    }
}
