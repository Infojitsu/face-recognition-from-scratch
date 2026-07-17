package svm.io;

import java.io.Serializable;
import svm.svm.SVM;

/**
 * Serializable container that groups:
 *  - the person's pseudonym
 *  - the SVM classifier trained for the person (person images = +1, the rest = -1)
 *  - the HOG vectors used for training (requirement: "the training vector data")
 *  - the corresponding labels
 *
 * Each person has one such file: classifiers/&lt;pseudonym&gt;.dat
 */
public class PersonClassifier implements Serializable {

    private static final long serialVersionUID = 1L;

    /** The person's pseudonym */
    public String pseudonym;
    /** The trained SVM classifier */
    public SVM svm;
    /** The HOG feature vectors used for training */
    public double[][] trainingFeatures;
    /** The corresponding labels (+1 / -1) */
    public double[] trainingLabels;

    /**
     * Constructor.
     */
    public PersonClassifier(String pseudonym, SVM svm,
                             double[][] trainingFeatures, double[] trainingLabels) {
        this.pseudonym = pseudonym;
        this.svm = svm;
        this.trainingFeatures = trainingFeatures;
        this.trainingLabels = trainingLabels;
    }
}
