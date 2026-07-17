package svm.svm;

import java.io.Serializable;

/**
 * Kernel function for the SVM. Must be serializable
 * because it is saved together with the trained classifier.
 */
public interface Kernel extends Serializable {

    /**
     * Computes K(x, y).
     */
    double apply(double[] x, double[] y);
}
