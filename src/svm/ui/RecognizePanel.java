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
 * Panou pentru cerinta (8): preia imagini de la webcam cu un anumit FPS,
 * detecteaza patratele cap, extrage HOG, trece prin fiecare clasificator
 * de persoana. Daca unul returneaza +1, scrie pseudonimul deasupra patratului.
 */
public class RecognizePanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final ImagePanel preview = new ImagePanel();
    private final JSpinner fpsSpinner =
            new JSpinner(new SpinnerNumberModel(10, 1, 30, 1));
    private final JSlider thresholdSlider = new JSlider(-30, 100, 25);
    private final JLabel thresholdLabel = new JLabel("Prag: 0.25");
    private final JButton btnStart = new JButton("Start");
    private final JButton btnStop  = new JButton("Stop");
    private final JLabel status = new JLabel("Gata.");

    private Thread worker;
    private volatile boolean running;
    private List<PersonClassifier> classifiers = new ArrayList<>();

    /** Istoric pentru smoothing temporal: afiseaza eticheta majoritara din
     *  ultimele SMOOTH_WINDOW frame-uri, nu eticheta frame-ului curent.
     *  Elimina flicker-ul cand scorurile SVM sunt apropiate intre persoane. */
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
                thresholdLabel.setText(String.format("Prag: %.2f", t));
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
            status.setText("Eroare incarcare clasificatoare: " + ex.getMessage());
            return;
        }
        if (classifiers.isEmpty()) {
            status.setText("Nu exista clasificatoare in " + AppContext.CLASSIFIERS_DIR);
            return;
        }
        btnStart.setEnabled(false);
        btnStop.setEnabled(true);
        running = true;
        final int fps = (Integer) fpsSpinner.getValue();
        status.setText("Recunoastere live | " + classifiers.size() + " clasificatoare");

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
            finishWithError("Sursa indisponibila: " + ex.getMessage());
            return;
        }
        try {
            while (running) {
                long t0 = System.currentTimeMillis();
                Image frame;
                try { frame = src.grab(); }
                catch (Exception ex) { finishWithError("Eroare captura: " + ex.getMessage()); return; }
                if (frame == null) break;

                // O singura parcurgere sliding window per frame; selectPrimary
                // actualizeaza tracker-ul si returneaza fallback-ul coasted
                // cand detect() e gol. (Inainte detect() era apelat de 2 ori.)
                List<Rect> heads = new ArrayList<>(det.detect(frame));
                Rect stabilized = det.selectPrimary(heads);
                if (stabilized != null && heads.isEmpty()) {
                    heads.add(stabilized);
                }

                List<String> labels = new ArrayList<>();
                for (Rect r : heads) {
                    // Protectie contra coordonate invalide dupa EMA
                    int x1 = Math.max(0, r.x);
                    int y1 = Math.max(0, r.y);
                    int x2 = Math.min(frame.getWidth() - 1, r.x + r.w - 1);
                    int y2 = Math.min(frame.getHeight() - 1, r.y + r.h - 1);
                    if (x2 - x1 < 30 || y2 - y1 < 30) { labels.add("?"); continue; }

                    Image crop = frame.crop(x1, y1, x2, y2).scale(HOG.IMG, HOG.IMG);
                    double[] feat = HOG.compute(crop);
                    // Urmarim primul SI al doilea scor: daca diferenta e mica,
                    // e ambiguu (probabil crop partial/neclar), afiseaza "?"
                    // in loc sa alegem arbitrar intre persoane apropiate la scor.
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
                    // Cere marja relativa de 25% fata de urmatorul scor pozitiv.
                    // Ex: best=0.10, second=0.09 -> raport 0.90 -> "?" (ambigu).
                    //     best=0.10, second=0.05 -> raport 0.50 -> label valid.
                    boolean ambiguous = best != null && secondScore > 0
                            && secondScore / bestScore > 0.75;
                    labels.add(best == null || ambiguous ? "?" : best);
                }

                // Smoothing temporal pe eticheta principala (indexul 0).
                // In loc sa afisam moda frame-ului curent (care poate flipa
                // rapid intre persoane cand scorurile sunt aproape), afisam
                // moda ultimelor SMOOTH_WINDOW frame-uri.
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

                // Dedupe: maxim 1 patrat per persoana etichetata.
                // Detectorul produce uneori 2-3 ferestre suprapuse sau
                // false positives labelate pe fundal. Pastram doar cea
                // cu scor head-detector maxim pentru fiecare eticheta
                // (heads e sortat descrescator, deci primul = maxim).
                // Eliminam "?"-urile complet - sunt zgomot vizual.
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
