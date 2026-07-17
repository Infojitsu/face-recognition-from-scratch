package svm.core;

import java.io.Serializable;

/**
 * Dreptunghi axis-aligned folosit pentru patratele cap detectate.
 * Reprezentat prin coordonate stanga-sus (x,y) si dimensiuni.
 */
public class Rect implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Coordonata X (stanga) */
    public int x;
    /** Coordonata Y (sus) */
    public int y;
    /** Latime */
    public int w;
    /** Inaltime */
    public int h;

    /** Constructor complet. */
    public Rect(int x, int y, int w, int h) {
        this.x = x; this.y = y; this.w = w; this.h = h;
    }

    /** @return aria dreptunghiului */
    public int area() { return w * h; }

    /** @return coordonata X a marginii drepte (exclusiv) */
    public int x2() { return x + w; }
    /** @return coordonata Y a marginii de jos (exclusiv) */
    public int y2() { return y + h; }

    @Override
    public String toString() {
        return "Rect[" + x + "," + y + "," + w + "x" + h + "]";
    }
}
