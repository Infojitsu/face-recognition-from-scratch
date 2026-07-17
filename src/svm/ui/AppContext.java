package svm.ui;

import java.io.File;
import svm.detector.HeadDetector;
import svm.svm.SVM;

/**
 * State shared between panels (simple singleton).
 * Holds the current head detector and the folder paths.
 */
public class AppContext {

    /** Face image folder (head-detector training) - subfolders positive/, negative/ */
    public static final File HEAD_IMAGES_DIR = new File("head_images");
    /** Per-person face image folder (pseudonym) */
    public static final File FACES_DIR       = new File("faces");
    /** Precomputed HOG vector folder (requirement 7: delivery) */
    public static final File HOG_VECTORS_DIR = new File("hog_vectors");
    /** Saved classifier folder */
    public static final File CLASSIFIERS_DIR = new File("classifiers");
    /** SVM head verifier file */
    public static final File HEAD_VERIFIER   = new File("classifiers/_head_verifier.dat");

    private static HeadDetector headDetector = new HeadDetector();

    /** @return the current detector (may be with or without a verifier) */
    public static HeadDetector getHeadDetector() { return headDetector; }

    /** Sets the SVM verifier on the current detector. */
    public static void setHeadVerifier(SVM v) {
        headDetector.setVerifier(v);
    }

    /** Initializes the folders if they do not exist. */
    public static void ensureDirs() {
        HEAD_IMAGES_DIR.mkdirs();
        new File(HEAD_IMAGES_DIR, "positive").mkdirs();
        new File(HEAD_IMAGES_DIR, "negative").mkdirs();
        FACES_DIR.mkdirs();
        HOG_VECTORS_DIR.mkdirs();
        CLASSIFIERS_DIR.mkdirs();
    }
}
