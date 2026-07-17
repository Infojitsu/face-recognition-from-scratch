package svm.ui;

import java.io.File;
import svm.detector.HeadDetector;
import svm.svm.SVM;

/**
 * Stare partajata intre panouri (singleton simplu).
 * Pastreaza detectorul de cap curent si caile catre foldere.
 */
public class AppContext {

    /** Folder imagini fete (antrenare detector-cap) - subfoldere positive/, negative/ */
    public static final File HEAD_IMAGES_DIR = new File("head_images");
    /** Folder imagini fete per persoana (pseudonim) */
    public static final File FACES_DIR       = new File("faces");
    /** Folder vectori HOG precalculati (cerinta 7: livrare) */
    public static final File HOG_VECTORS_DIR = new File("hog_vectors");
    /** Folder clasificatoare salvate */
    public static final File CLASSIFIERS_DIR = new File("classifiers");
    /** Fisier verificator SVM cap */
    public static final File HEAD_VERIFIER   = new File("classifiers/_head_verifier.dat");

    private static HeadDetector headDetector = new HeadDetector();

    /** @return detectorul curent (poate fi cu sau fara verificator) */
    public static HeadDetector getHeadDetector() { return headDetector; }

    /** Seteaza verificatorul SVM in detectorul curent. */
    public static void setHeadVerifier(SVM v) {
        headDetector.setVerifier(v);
    }

    /** Initializeaza folderele daca nu exista. */
    public static void ensureDirs() {
        HEAD_IMAGES_DIR.mkdirs();
        new File(HEAD_IMAGES_DIR, "positive").mkdirs();
        new File(HEAD_IMAGES_DIR, "negative").mkdirs();
        FACES_DIR.mkdirs();
        HOG_VECTORS_DIR.mkdirs();
        CLASSIFIERS_DIR.mkdirs();
    }
}
