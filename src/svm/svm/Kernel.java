package svm.svm;

import java.io.Serializable;

/**
 * Functie nucleu (kernel) pentru SVM. Trebuie sa fie serializabila
 * pentru ca va fi salvata impreuna cu clasificatorul antrenat.
 */
public interface Kernel extends Serializable {

    /**
     * Calculeaza K(x, y).
     */
    double apply(double[] x, double[] y);
}
