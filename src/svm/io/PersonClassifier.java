package svm.io;

import java.io.Serializable;
import svm.svm.SVM;

/**
 * Container serializabil care grupeaza:
 *  - pseudonimul persoanei
 *  - clasificatorul SVM antrenat pentru persoana (imagini pers = +1, restul = -1)
 *  - vectorii HOG folositi la antrenare (cerinta: "datele vectorilor de invatare")
 *  - etichetele corespunzatoare
 *
 * Fiecare persoana are cate un astfel de fisier: classifiers/&lt;pseudonim&gt;.dat
 */
public class PersonClassifier implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Pseudonimul persoanei */
    public String pseudonym;
    /** Clasificatorul SVM antrenat */
    public SVM svm;
    /** Vectorii de trasaturi HOG folositi la antrenare */
    public double[][] trainingFeatures;
    /** Etichetele corespunzatoare (+1 / -1) */
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
