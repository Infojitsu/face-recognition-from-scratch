package svm.core;

import java.io.Serializable;

/**
 * Axis-aligned rectangle used for the detected head squares.
 * Represented by top-left coordinates (x,y) and dimensions.
 */
public class Rect implements Serializable {

    private static final long serialVersionUID = 1L;

    /** X coordinate (left) */
    public int x;
    /** Y coordinate (top) */
    public int y;
    /** Width */
    public int w;
    /** Height */
    public int h;

    /** Full constructor. */
    public Rect(int x, int y, int w, int h) {
        this.x = x; this.y = y; this.w = w; this.h = h;
    }

    /** @return the area of the rectangle */
    public int area() { return w * h; }

    /** @return the X coordinate of the right edge (exclusive) */
    public int x2() { return x + w; }
    /** @return the Y coordinate of the bottom edge (exclusive) */
    public int y2() { return y + h; }

    @Override
    public String toString() {
        return "Rect[" + x + "," + y + "," + w + "x" + h + "]";
    }
}
