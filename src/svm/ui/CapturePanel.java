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
 * Panou pentru cerinta (3): preia N imagini de la webcam, detecteaza capul
 * cel mai mare, scaleaza la 128x128 si salveaza in faces/&lt;pseudonim&gt;/.
 * Numele fisierelor: pseudonim_yyyyMMdd_HHmmssSSS.png
 */
public class CapturePanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final ImagePanel preview = new ImagePanel();
    private final JTextField pseudoField = new JTextField("persoana1", 14);
    private final JSpinner countSpinner =
            new JSpinner(new SpinnerNumberModel(500, 10, 5000, 10));
    private final JSlider thresholdSlider = new JSlider(-30, 100, 25);
    private final JLabel thresholdLabel = new JLabel("Prag: 0.25");
    private final JButton btnStart = new JButton("Start capturare");
    private final JButton btnStop  = new JButton("Stop");
    private final JLabel status = new JLabel("Gata.");
    private final JProgressBar progress = new JProgressBar(0, 100);

    private Thread worker;
    private volatile boolean running;

    /** Cat de mult peste pragul detectorului trebuie sa fie scorul ca sa
     *  salvam. Respinge detectiile marginale (porneste 0.15 peste slider). */
    private static final double SAVE_SCORE_MARGIN = 0.15;
    /** Interval minim intre salvari (ms). Evita 10 poze identice pe secunda
     *  si permite subiectului sa-si schimbe pozitia/expresia intre save-uri. */
    private static final long MIN_SAVE_INTERVAL_MS = 100;
    /** Prag minim pe varianta Laplacian a crop-ului - respinge motion blur
     *  si frame-urile out-of-focus (fetele clare au >500, blur-ul <200). */
    private static final double MIN_SHARPNESS = 200.0;

    public CapturePanel() {
        super(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JPanel top = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 4));
        top.add(new JLabel("Pseudonim:"));
        top.add(pseudoField);
        top.add(new JLabel("Nr. imagini:"));
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
                thresholdLabel.setText(String.format("Prag: %.2f", t));
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
            JOptionPane.showMessageDialog(this, "Introdu un pseudonim.");
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
            finishWithError("Sursa indisponibila: " + ex.getMessage());
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
                catch (Exception ex) { finishWithError("Eroare captura: " + ex.getMessage()); return; }
                if (frame == null) break;

                // Apel detect() direct (nu largestHeadRect) ca sa avem scorul SVM.
                List<Rect> heads = det.detect(frame);
                Rect head = det.selectPrimary(heads);
                double score = heads.isEmpty() ? 0.0
                        : det.getLastDetectScores().get(0);

                // Preview cu feedback: label arata scorul SVM - operatorul
                // vede live daca detectia e "ferma" (0.6+) sau "marginala".
                preview.setImage(frame.toBufferedImage());
                List<Rect> rs = new ArrayList<>();
                List<String> ls = new ArrayList<>();
                if (head != null) {
                    rs.add(head);
                    ls.add(heads.isEmpty()
                            ? "coast"
                            : String.format("cap %.2f", score));
                }
                preview.setRects(rs, ls);

                // === GATE-URI DE CALITATE (toate trebuie satisfacute ca sa salvam) ===

                // Gate 0: detectie reala (nu din coasting - scorul e zero acolo)
                if (head == null || heads.isEmpty()) {
                    try { Thread.sleep(30); } catch (InterruptedException ignored) {}
                    continue;
                }

                // Gate 1: scor mult peste prag (respinge detectiile marginale)
                double minScoreToSave = det.getThreshold() + SAVE_SCORE_MARGIN;
                if (score < minScoreToSave) {
                    skippedLowScore++;
                    try { Thread.sleep(30); } catch (InterruptedException ignored) {}
                    continue;
                }

                // Gate 2: minim 100ms de la ultima salvare (evita duplicate)
                long now = System.currentTimeMillis();
                if (now - lastSaveMs < MIN_SAVE_INTERVAL_MS) {
                    skippedTooSoon++;
                    try { Thread.sleep(30); } catch (InterruptedException ignored) {}
                    continue;
                }

                // Gate 3: claritate (Laplacian variance pe crop-ul original,
                // nu pe cel scalat la 128x128 - scalarea ascunde motion blur).
                Image origCrop = frame.crop(head.x, head.y,
                        head.x + head.w - 1, head.y + head.h - 1);
                double[][] origGray = origCrop.toGrayscale();
                double sharp = laplacianVariance(origGray);
                if (sharp < MIN_SHARPNESS) {
                    skippedBlurry++;
                    try { Thread.sleep(30); } catch (InterruptedException ignored) {}
                    continue;
                }

                // Toate gate-urile trecute: salvam
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
                                "Salvate %d/%d | scor %.2f | claritate %.0f | skip: scor %d, blur %d",
                                s, target, fScore, fSharp, fLow, fBlur));
                        }
                    });
                } catch (Exception e) {
                    // ignora o imagine cu eroare de I/O, continua
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
     * Varianta operatorului Laplacian pe imagine grayscale - metrica
     * standard pentru sharpness. Valori mari = imagine clara, valori mici
     * = out-of-focus sau motion blur. L(x,y) = 4*I - I_sus - I_jos - I_st - I_dr.
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
