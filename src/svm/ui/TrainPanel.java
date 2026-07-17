package svm.ui;

import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import javax.imageio.ImageIO;
import javax.swing.*;
import svm.core.Image;
import svm.detector.TrainingPipeline;
import svm.hog.HOG;
import svm.io.Storage;
import svm.svm.SVM;

/**
 * Panel for:
 *   (5) HOG vector extraction
 *   (6) training one SVM classifier per person
 *   (7) SMO with Sigmoid kernel (default in TrainingPipeline)
 *   + optional: head verifier training (requirement 1)
 *
 * Shows a text log with the training progress.
 */
public class TrainPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final JButton btnTrainHead = new JButton("1. Train head detector (SVM verifier)");
    private final JButton btnLoadHead  = new JButton("Load head detector");
    private final JButton btnExtractHog = new JButton("2. Extract HOG vectors (saved in hog_vectors/)");
    private final JButton btnTrainAll  = new JButton("3. Train person classifiers (SMO + Sigmoid)");
    private final JTextArea log = new JTextArea();

    public TrainPanel() {
        super(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JPanel top = new JPanel(new GridLayout(2, 2, 6, 6));
        top.add(btnTrainHead);
        top.add(btnLoadHead);
        top.add(btnExtractHog);
        top.add(btnTrainAll);
        add(top, BorderLayout.NORTH);

        log.setEditable(false);
        log.setFont(new Font("Monospaced", Font.PLAIN, 12));
        add(new JScrollPane(log), BorderLayout.CENTER);

        // Redirect System.out to the text area
        redirectOut();

        btnTrainHead.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) {
                runTask(new Runnable() { @Override public void run() { trainHead(); } });
            }
        });
        btnLoadHead.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) {
                runTask(new Runnable() { @Override public void run() { loadHead(); } });
            }
        });
        btnExtractHog.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) {
                runTask(new Runnable() { @Override public void run() { extractHog(); } });
            }
        });
        btnTrainAll.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) {
                runTask(new Runnable() { @Override public void run() { trainAll(); } });
            }
        });
    }

    private void redirectOut() {
        PrintStream ps = new PrintStream(new java.io.OutputStream() {
            @Override public void write(final int b) throws IOException {
                SwingUtilities.invokeLater(new Runnable() {
                    @Override public void run() { log.append(String.valueOf((char) b)); }
                });
            }
            @Override public void write(byte[] buf, int off, int len) {
                final String s = new String(buf, off, len);
                SwingUtilities.invokeLater(new Runnable() {
                    @Override public void run() { log.append(s); }
                });
            }
        }, true);
        System.setOut(ps);
    }

    private void runTask(final Runnable r) {
        new Thread(new Runnable() {
            @Override public void run() {
                setButtons(false);
                try { r.run(); }
                catch (Throwable t) {
                    System.out.println("ERROR: " + t.getMessage());
                    t.printStackTrace(System.out);
                } finally { setButtons(true); }
            }
        }, "train-worker").start();
    }

    private void setButtons(final boolean enabled) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override public void run() {
                btnTrainHead.setEnabled(enabled);
                btnLoadHead.setEnabled(enabled);
                btnExtractHog.setEnabled(enabled);
                btnTrainAll.setEnabled(enabled);
            }
        });
    }

    // ------- Tasks -------

    private void trainHead() {
        System.out.println("== Training head detector (Linear kernel) ==");
        File pos = new File(AppContext.HEAD_IMAGES_DIR, "positive");
        File neg = new File(AppContext.HEAD_IMAGES_DIR, "negative");
        if (pos.listFiles() == null || pos.listFiles().length == 0) {
            System.out.println("There are no images in " + pos.getAbsolutePath());
            System.out.println("Add images with a head (jpg/png) and without a head in negative/, then retry.");
            return;
        }
        try {
            SVM svm = TrainingPipeline.trainHeadVerifier(pos, neg);
            Storage.save(svm, AppContext.HEAD_VERIFIER);
            AppContext.setHeadVerifier(svm);
            System.out.println("OK. Support vectors: " + svm.numSupportVectors());
            System.out.println("Saved to " + AppContext.HEAD_VERIFIER.getAbsolutePath());
        } catch (Exception ex) {
            System.out.println("Training failed: " + ex.getMessage());
        }
    }

    private void loadHead() {
        if (!AppContext.HEAD_VERIFIER.exists()) {
            System.out.println("File does not exist: " + AppContext.HEAD_VERIFIER);
            return;
        }
        try {
            SVM svm = Storage.load(AppContext.HEAD_VERIFIER);
            AppContext.setHeadVerifier(svm);
            System.out.println("Head detector loaded (SV=" + svm.numSupportVectors() + ")");
        } catch (Exception ex) {
            System.out.println("Loading failed: " + ex.getMessage());
        }
    }

    private void extractHog() {
        System.out.println("== Extracting HOG vectors from faces/ -> hog_vectors/ ==");
        File[] subs = AppContext.FACES_DIR.listFiles(File::isDirectory);
        if (subs == null || subs.length == 0) {
            System.out.println("There are no subfolders in faces/");
            return;
        }
        for (File sub : subs) {
            File outDir = new File(AppContext.HOG_VECTORS_DIR, sub.getName());
            outDir.mkdirs();
            File[] imgs = sub.listFiles((f, n) -> n.toLowerCase().endsWith(".png")
                                              || n.toLowerCase().endsWith(".jpg"));
            if (imgs == null) continue;
            int ok = 0;
            for (File f : imgs) {
                try {
                    Image im = new Image(ImageIO.read(f)).scale(HOG.IMG, HOG.IMG);
                    double[] v = HOG.compute(im);
                    File out = new File(outDir, f.getName().replaceAll("\\.[^.]+$", ".hog"));
                    try (java.io.DataOutputStream dos = new java.io.DataOutputStream(
                            new java.io.BufferedOutputStream(new java.io.FileOutputStream(out)))) {
                        dos.writeInt(v.length);
                        for (double d : v) dos.writeDouble(d);
                    }
                    ok++;
                } catch (Exception e) { /* skip */ }
            }
            System.out.println("  " + sub.getName() + ": " + ok + " vectors");
        }
        System.out.println("HOG extraction done.");
    }

    private void trainAll() {
        System.out.println("== Training one SVM per person (SMO + Sigmoid) ==");
        try {
            TrainingPipeline.trainPersonClassifiers(
                    AppContext.FACES_DIR, AppContext.CLASSIFIERS_DIR);
            System.out.println("Done. Classifiers saved in " + AppContext.CLASSIFIERS_DIR);
        } catch (Exception ex) {
            System.out.println("Failed: " + ex.getMessage());
        }
    }
}
