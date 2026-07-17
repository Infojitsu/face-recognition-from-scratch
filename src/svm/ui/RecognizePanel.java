package svm.ui;

import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import javax.swing.*;
import svm.core.Image;
import svm.core.Rect;
import svm.detector.HeadDetector;
import svm.detector.TrainingPipeline;
import svm.hog.HOG;
import svm.io.PersonClassifier;
import svm.util.FolderSource;
import svm.util.ImageSource;
import svm.util.WebcamSource;

/**
 * Panel for requirement (8): grabs images from the webcam at a given FPS,
 * detects the head squares, extracts HOG, runs it through every person
 * classifier. If one returns +1, writes the pseudonym above the square.
 */
public class RecognizePanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final ImagePanel preview = new ImagePanel();
    private final JSpinner fpsSpinner =
            new JSpinner(new SpinnerNumberModel(10, 1, 30, 1));
    private final JSlider thresholdSlider = new JSlider(-30, 100, 25);
    private final JLabel thresholdLabel = new JLabel("Threshold: 0.25");
    private final JButton btnStart = new JButton("Start");
    private final JButton btnStop  = new JButton("Stop");
    private final JLabel status = new JLabel("Ready.");

    private Thread worker;
    private volatile boolean running;
    private List<PersonClassifier> classifiers = new ArrayList<>();

    /** History for temporal smoothing: show the majority label from the
     *  last SMOOTH_WINDOW frames, not the current frame's label.
     *  Removes flicker when SVM scores are close between persons. */
    private final java.util.Deque<String> labelHistory = new java.util.ArrayDeque<String>();
    private static final int SMOOTH_WINDOW = 5;

    public RecognizePanel() {
        super(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JPanel top = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 4));
        top.add(new JLabel("FPS:"));
        top.add(fpsSpinner);
        top.add(thresholdLabel);
        thresholdSlider.setPreferredSize(new Dimension(150, 20));
        top.add(thresholdSlider);
        top.add(btnStart);
        top.add(btnStop);
        btnStop.setEnabled(false);

        add(top, BorderLayout.NORTH);
        add(preview, BorderLayout.CENTER);
        add(status, BorderLayout.SOUTH);

        thresholdSlider.addChangeListener(new javax.swing.event.ChangeListener() {
            @Override public void stateChanged(javax.swing.event.ChangeEvent e) {
                double t = thresholdSlider.getValue() / 100.0;
                thresholdLabel.setText(String.format("Threshold: %.2f", t));
                AppContext.getHeadDetector().setThreshold(t);
            }
        });

        btnStart.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) { start(); }
        });
        btnStop.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) { running = false; }
        });
    }

    private void start() {
        try {
            classifiers = TrainingPipeline.loadAll(AppContext.CLASSIFIERS_DIR);
        } catch (Exception ex) {
            status.setText("Error loading classifiers: " + ex.getMessage());
            return;
        }
        if (classifiers.isEmpty()) {
            status.setText("There are no classifiers in " + AppContext.CLASSIFIERS_DIR);
            return;
        }
        btnStart.setEnabled(false);
        btnStop.setEnabled(true);
        running = true;
        final int fps = (Integer) fpsSpinner.getValue();
        status.setText("Live recognition | " + classifiers.size() + " classifiers");

        worker = new Thread(new Runnable() {
            @Override public void run() { loop(fps); }
        }, "recognize-worker");
        worker.start();
    }

    private void loop(int fps) {
        long period = 1000L / fps;
        HeadDetector det = AppContext.getHeadDetector();
        det.setTrackingEnabled(true);
        det.resetTracking();
        labelHistory.clear();
        ImageSource src = new WebcamSource();

        try {
            src.open();
        } catch (Exception ex) {
            finishWithError("Source unavailable: " + ex.getMessage());
            return;
        }
        try {
            while (running) {
                long t0 = System.currentTimeMillis();
                Image frame;
                try { frame = src.grab(); }
                catch (Exception ex) { finishWithError("Capture error: " + ex.getMessage()); return; }
                if (frame == null) break;

                // A single sliding-window pass per frame; selectPrimary
                // updates the tracker and returns the coasted fallback
                // when detect() is empty. (Previously detect() was called twice.)
                List<Rect> heads = new ArrayList<>(det.detect(frame));
                Rect stabilized = det.selectPrimary(heads);
                if (stabilized != null && heads.isEmpty()) {
                    heads.add(stabilized);
                }

                List<String> labels = new ArrayList<>();
                for (Rect r : heads) {
                    // Guard against invalid coordinates after EMA
                    int x1 = Math.max(0, r.x);
                    int y1 = Math.max(0, r.y);
                    int x2 = Math.min(frame.getWidth() - 1, r.x + r.w - 1);
                    int y2 = Math.min(frame.getHeight() - 1, r.y + r.h - 1);
                    if (x2 - x1 < 30 || y2 - y1 < 30) { labels.add("?"); continue; }

                    Image crop = frame.crop(x1, y1, x2, y2).scale(HOG.IMG, HOG.IMG);
                    double[] feat = HOG.compute(crop);
                    // Track the first AND second score: if the difference is small,
                    // it's ambiguous (probably a partial/unclear crop), show "?"
                    // instead of arbitrarily choosing between close-scoring persons.
                    String best = null;
                    double bestScore = 0;
                    double secondScore = 0;
                    for (PersonClassifier pc : classifiers) {
                        double s = pc.svm.decisionFunction(feat);
                        if (s <= 0) continue;
                        if (s > bestScore) {
                            secondScore = bestScore;
                            bestScore = s;
                            best = pc.pseudonym;
                        } else if (s > secondScore) {
                            secondScore = s;
                        }
                    }
                    // Require a 25% relative margin over the next positive score.
                    // E.g.: best=0.10, second=0.09 -> ratio 0.90 -> "?" (ambiguous).
                    //       best=0.10, second=0.05 -> ratio 0.50 -> valid label.
                    boolean ambiguous = best != null && secondScore > 0
                            && secondScore / bestScore > 0.75;
                    labels.add(best == null || ambiguous ? "?" : best);
                }

                // Temporal smoothing on the primary label (index 0).
                // Instead of showing the current frame's pick (which can flip
                // quickly between persons when scores are close), show the
                // mode of the last SMOOTH_WINDOW frames.
                if (!labels.isEmpty()) {
                    labelHistory.addLast(labels.get(0));
                    while (labelHistory.size() > SMOOTH_WINDOW) labelHistory.removeFirst();
                    java.util.Map<String, Integer> counts = new java.util.HashMap<String, Integer>();
                    for (String l : labelHistory) {
                        Integer c = counts.get(l);
                        counts.put(l, c == null ? 1 : c + 1);
                    }
                    String mode = labels.get(0);
                    int bestCount = 0;
                    for (java.util.Map.Entry<String, Integer> e : counts.entrySet()) {
                        if (e.getValue() > bestCount) {
                            bestCount = e.getValue();
                            mode = e.getKey();
                        }
                    }
                    labels.set(0, mode);
                }

                // Dedupe: at most 1 square per labeled person.
                // The detector sometimes produces 2-3 overlapping windows or
                // labeled false positives on the background. We keep only the
                // one with the maximum head-detector score for each label
                // (heads is sorted descending, so the first = maximum).
                // Remove the "?" entries entirely - they are visual noise.
                java.util.Map<String, Integer> firstIdx =
                        new java.util.LinkedHashMap<String, Integer>();
                for (int i = 0; i < heads.size(); i++) {
                    String lbl = labels.get(i);
                    if ("?".equals(lbl)) continue;
                    if (!firstIdx.containsKey(lbl)) firstIdx.put(lbl, i);
                }
                List<Rect> dedupHeads = new ArrayList<>();
                List<String> dedupLabels = new ArrayList<>();
                for (Integer idx : firstIdx.values()) {
                    dedupHeads.add(heads.get(idx));
                    dedupLabels.add(labels.get(idx));
                }
                heads = dedupHeads;
                labels = dedupLabels;

                preview.setImage(frame.toBufferedImage());
                preview.setRects(heads, labels);

                long elapsed = System.currentTimeMillis() - t0;
                long sleep = period - elapsed;
                if (sleep > 0) try { Thread.sleep(sleep); } catch (InterruptedException ignored) {}
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
