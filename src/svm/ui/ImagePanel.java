package svm.ui;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JPanel;
import svm.core.Rect;

/**
 * Swing component that draws an image + rectangles on top of it.
 * Used in all panels for webcam / detection display.
 */
public class ImagePanel extends JPanel {

    private static final long serialVersionUID = 1L;

    /** The current image */
    private BufferedImage image;
    /** The list of rectangles to draw (with optional labels) */
    private final List<Rect> rects = new ArrayList<>();
    /** Labels for the rectangles (same index) */
    private final List<String> labels = new ArrayList<>();

    public ImagePanel() {
        setBackground(new Color(24, 26, 30));
    }

    /** Updates the displayed image. */
    public synchronized void setImage(BufferedImage img) {
        this.image = img;
        repaint();
    }

    /** Updates the list of rectangles + labels. */
    public synchronized void setRects(List<Rect> rs, List<String> lbls) {
        rects.clear(); labels.clear();
        if (rs != null) rects.addAll(rs);
        if (lbls != null) labels.addAll(lbls);
        repaint();
    }

    /** Clears the panel (image + rectangles). */
    public synchronized void clear() {
        image = null;
        rects.clear();
        labels.clear();
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (image == null) return;
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                             RenderingHints.VALUE_INTERPOLATION_BILINEAR);

        // Fit the image while keeping the aspect ratio
        int pw = getWidth(), ph = getHeight();
        double ir = (double) image.getWidth() / image.getHeight();
        double pr = (double) pw / ph;
        int dw, dh;
        if (ir > pr) { dw = pw; dh = (int)(pw / ir); }
        else         { dh = ph; dw = (int)(ph * ir); }
        int dx = (pw - dw) / 2;
        int dy = (ph - dh) / 2;
        g2.drawImage(image, dx, dy, dw, dh, null);

        // Draw the scaled rectangles
        double sx = (double) dw / image.getWidth();
        double sy = (double) dh / image.getHeight();
        g2.setStroke(new BasicStroke(3f));
        g2.setColor(new Color(40, 220, 90));
        g2.setFont(new Font("SansSerif", Font.BOLD, 16));
        synchronized (this) {
            for (int i = 0; i < rects.size(); i++) {
                Rect r = rects.get(i);
                int rx = dx + (int)(r.x * sx);
                int ry = dy + (int)(r.y * sy);
                int rw = (int)(r.w * sx);
                int rh = (int)(r.h * sy);
                g2.drawRect(rx, ry, rw, rh);
                String lbl = i < labels.size() ? labels.get(i) : null;
                if (lbl != null && !lbl.isEmpty()) {
                    FontMetrics fm = g2.getFontMetrics();
                    int tw = fm.stringWidth(lbl) + 10;
                    int th = fm.getHeight() + 4;
                    g2.fillRect(rx, ry - th, tw, th);
                    g2.setColor(Color.BLACK);
                    g2.drawString(lbl, rx + 5, ry - 5);
                    g2.setColor(new Color(40, 220, 90));
                }
            }
        }
        g2.dispose();
    }
}
