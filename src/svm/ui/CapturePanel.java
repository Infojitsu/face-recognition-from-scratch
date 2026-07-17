package svm.ui;

import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;
import javax.swing.*;
import svm.core.Image;
import svm.core.Rect;
import svm.detector.HeadDetector;
import svm.util.FolderSource;
import svm.util.ImageSource;
import svm.util.WebcamSource;

/**
 * Panel for requirement (3): grabs N images from the webcam, detects the
 * largest head, scales it to 128x128 and saves it in faces/&lt;pseudonym&gt;/.
 * File names: pseudonym_yyyyMMdd_HHmmssSSS.png
 */
public class CapturePanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final ImagePanel preview = new ImagePanel();
    private final JTextField pseudoField = new JTextField("person1", 14);
    private final JSpinner countSpinner =
            new JSpinner(new SpinnerNumberModel(500, 10, 5000, 10));
    private final JSlider thresholdSlider = new JSlider(-30, 100, 25);
    private final JLabel thresholdLabel = new JLabel("Threshold: 0.25");
    private final JButton btnStart = new JButton("Start capture");
    private final JButton btnStop  = new JButton("Stop");
    private final JLabel status = new JLabel("Ready.");
    private final JProgressBar progress = new JProgressBar(0, 100);

    private Thread worker;
    private volatile boolean running;

    /** How far above the detector threshold the score must be for us to
     *  save. Rejects marginal detections (starts 0.15 above the slider). */
    private static final double SAVE_SCORE_MARGIN = 0.15;
    /** Minimum interval between saves (ms). Avoids 10 identical photos per second
     *  and lets the subject change position/expression between saves. */
    private static final long MIN_SAVE_INTERVAL_MS = 100;
    /** Minimum threshold on the Laplacian variance of the crop - rejects motion blur
     *  and out-of-focus frames (sharp faces score >500, blur <200). */
    private static final double MIN_SHARPNESS = 200.0;

    public CapturePanel() {
        super(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JPanel top = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 4));
        top.add(new JLabel("Pseudonym:"));
        top.add(pseudoField);
        top.add(new JLabel("No. of images:"));
        top.add(countSpinner);
        top.add(thresholdLabel);
        thresholdSlider.setPreferredSize(new Dimension(150, 20));
        top.add(thresholdSlider);
        top.add(btnStart);
        top.add(btnStop);
        btnStop.setEnabled(false);

        thresholdSlider.addChangeListener(new javax.swing.event.ChangeListener() {
            @Override public void stateChanged(javax.swing.event.ChangeEvent e) {
                double t = thresholdSlider.getValue() / 100.0;
                thresholdLabel.setText(String.format("Threshold: %.2f", t));
                AppContext.getHeadDetector().setThreshold(t);
            }
        });

        JPanel bot = new JPanel(new BorderLayout(6, 6));
        bot.add(progress, BorderLayout.CENTER);
        bot.add(status, BorderLayout.SOUTH);

        add(top, BorderLayout.NORTH);
        add(preview, BorderLayout.CENTER);
        add(bot, BorderLayout.SOUTH);

        btnStart.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) { start(); }
        });
        btnStop.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) { running = false; }
        });
    }

    private void start() {
        final String pseudo = pseudoField.getText().trim();
        if (pseudo.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Enter a pseudonym.");
            return;
        }
        final int target = (Integer) countSpinner.getValue();

        btnStart.setEnabled(false);
        btnStop.setEnabled(true);
        progress.setMaximum(target);
        progress.setValue(0);
        running = true;

        worker = new Thread(new Runnable() {
            @Override public void run() { capture(pseudo, target); }
        }, "capture-worker");
        worker.start();
    }

    private void capture(String pseudo, int target) {
        File outDir = new File(AppContext.FACES_DIR, pseudo);
        outDir.mkdirs();
        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmssSSS");
        HeadDetector det = AppContext.getHeadDetector();
        det.setTrackingEnabled(true);
        det.resetTracking();

        ImageSource src = new WebcamSource();
        try {
            src.open();
        } catch (Exception ex) {
            finishWithError("Source unavailable: " + ex.getMessage());
            return;
        }

        int saved = 0;
        long lastSaveMs = 0;
        int skippedLowScore = 0;
        int skippedBlurry   = 0;
        int skippedTooSoon  = 0;
        try {
            while (running && saved < target) {
                Image frame;
                try { frame = src.grab(); }
                catch (Exception ex) { finishWithError("Capture error: " + ex.getMessage()); return; }
                if (frame == null) break;

                // Call detect() directly (not largestHeadRect) so we have the SVM score.
                List<Rect> heads = det.detect(frame);
                Rect head = det.selectPrimary(heads);
                double score = heads.isEmpty() ? 0.0
                        : det.getLastDetectScores().get(0);

                // Preview with feedback: the label shows the SVM score - the operator
                // sees live whether the detection is "firm" (0.6+) or "marginal".
                preview.setImage(frame.toBufferedImage());
                List<Rect> rs = new ArrayList<>();
                List<String> ls = new ArrayList<>();
                if (head != null) {
                    rs.add(head);
                    ls.add(heads.isEmpty()
                            ? "coast"
                            : String.format("head %.2f", score));
                }
                preview.setRects(rs, ls);

                // === QUALITY GATES (all must be satisfied for us to save) ===

                // Gate 0: real detection (not from coasting - the score is zero there)
                if (head == null || heads.isEmpty()) {
                    try { Thread.sleep(30); } catch (InterruptedException ignored) {}
                    continue;
                }

                // Gate 1: score well above the threshold (rejects marginal detections)
                double minScoreToSave = det.getThreshold() + SAVE_SCORE_MARGIN;
                if (score < minScoreToSave) {
                    skippedLowScore++;
                    try { Thread.sleep(30); } catch (InterruptedException ignored) {}
                    continue;
                }

                // Gate 2: at least 100ms since the last save (avoids duplicates)
                long now = System.currentTimeMillis();
                if (now - lastSaveMs < MIN_SAVE_INTERVAL_MS) {
                    skippedTooSoon++;
                    try { Thread.sleep(30); } catch (InterruptedException ignored) {}
                    continue;
                }

                // Gate 3: sharpness (Laplacian variance on the original crop,
                // not the one scaled to 128x128 - scaling hides motion blur).
                Image origCrop = frame.crop(head.x, head.y,
                        head.x + head.w - 1, head.y + head.h - 1);
                double[][] origGray = origCrop.toGrayscale();
                double sharp = laplacianVariance(origGray);
                if (sharp < MIN_SHARPNESS) {
                    skippedBlurry++;
                    try { Thread.sleep(30); } catch (InterruptedException ignored) {}
                    continue;
                }

                // All gates passed: save
                Image crop = origCrop.scale(128, 128);
                String fname = pseudo + "_" + sdf.format(new Date()) + ".png";
                try {
                    crop.saveAsPNG(new File(outDir, fname));
                    saved++;
                    lastSaveMs = now;
                    final int s = saved;
                    final double fScore = score;
                    final double fSharp = sharp;
                    final int fLow = skippedLowScore;
                    final int fBlur = skippedBlurry;
                    SwingUtilities.invokeLater(new Runnable() {
                        @Override public void run() {
                            progress.setValue(s);
                            status.setText(String.format(
                                "Saved %d/%d | score %.2f | sharpness %.0f | skip: score %d, blur %d",
                                s, target, fScore, fSharp, fLow, fBlur));
                        }
                    });
                } catch (Exception e) {
                    // ignore an image with an I/O error, continue
                }
                try { Thread.sleep(30); } catch (InterruptedException ignored) {}
            }
        } finally {
            src.close();
            det.setTrackingEnabled(false);
            preview.clear();
            SwingUtilities.invokeLater(new Runnable() {
                @Override public void run() {
                    btnStart.setEnabled(true);
                    btnStop.setEnabled(false);
                    running = false;
                }
            });
        }
    }

    /**
     * Variance of the Laplacian operator on a grayscale image - the standard
     * sharpness metric. High values = sharp image, low values
     * = out-of-focus or motion blur. L(x,y) = 4*I - I_up - I_down - I_left - I_right.
     */
    private static double laplacianVariance(double[][] gray) {
        int h = gray.length;
        int w = gray[0].length;
        if (h < 3 || w < 3) return 0.0;
        double sum = 0, sum2 = 0;
        int n = 0;
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                double l = 4.0 * gray[y][x]
                        - gray[y - 1][x] - gray[y + 1][x]
                        - gray[y][x - 1] - gray[y][x + 1];
                sum  += l;
                sum2 += l * l;
                n++;
            }
        }
        double mean = sum / n;
        return sum2 / n - mean * mean;
    }

    private void finishWithError(final String msg) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override public void run() {
                status.setText(msg);
                btnStart.setEnabled(true);
                btnStop.setEnabled(false);
                running = false;
            }
        });
    }
}
