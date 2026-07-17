package svm.svm;

import java.io.Serializable;
import java.util.Random;

/**
 * Clasificator SVM binar (+1/-1) antrenat cu algoritmul SMO
 * (Sequential Minimal Optimization), Platt 1998.
 *
 * Rezolva problema duala:
 *   max    sum_i alpha_i - 0.5 * sum_ij alpha_i alpha_j y_i y_j K(x_i, x_j)
 *   s.t.   0 &lt;= alpha_i &lt;= C
 *          sum_i alpha_i y_i = 0
 *
 * Decizia: f(x) = sum_i alpha_i y_i K(x_i, x) - b
 * Clasa: sign(f(x))
 *
 * La salvare prin serializare, pastram doar vectorii suport (alpha_i > 0)
 * pentru a minimiza dimensiunea fisierelor.
 */
public class SVM implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Kernelul folosit (Sigmoid implicit) */
    private Kernel kernel;
    /** Vectorii suport (dupa antrenare) */
    private double[][] supportX;
    /** Etichetele vectorilor suport */
    private double[] supportY;
    /** Multiplicatorii Lagrange pentru vectorii suport */
    private double[] supportA;
    /** Termenul bias (b) */
    private double b;
    /** Parametrul de regularizare C */
    private double C;
    /** Tolerante pentru convergenta */
    private double tol;
    /** Numar maxim de iteratii fara modificari */
    private int maxPasses;
    /** Cache pentru kernel liniar: w = sum(alpha_i * y_i * x_i).
     *  Transient - se reconstruieste la cerere dupa deserializare.
     *  Colapseaza f(x) din O(N*D) (peste N support vectors) in O(D) (un singur dot).
     */
    private transient double[] wLinear;

    /**
     * Constructor cu kernel si parametri SMO.
     * @param kernel functia nucleu
     * @param C regularizare (tipic 1.0)
     * @param tol toleranta (tipic 1e-3)
     * @param maxPasses iteratii fara schimbari pana la oprire (tipic 10)
     */
    public SVM(Kernel kernel, double C, double tol, int maxPasses) {
        this.kernel = kernel;
        this.C = C;
        this.tol = tol;
        this.maxPasses = maxPasses;
    }

    /**
     * Antreneaza clasificatorul pe setul (X, y) cu y in {-1, +1}.
     * Implementeaza algoritmul SMO simplificat Platt.
     */
    /**
     * Pentru kernel liniar, nu e nevoie de matrice kernel (O(n^2) memorie)
     * - putem tine direct vectorul w = sum(alpha_i * y_i * x_i) in O(n*d).
     * K(x_i, x_j) = x_i . x_j se calculeaza pe loc in O(d).
     *
     * Rezultat: memorie si timp dramatic mai mici pentru n mare.
     * Ex: n=3000, d=8100:
     *   - cache full: 72 MB memorie, ~30 min timp
     *   - fast linear: ~200 KB memorie extra, ~2 min timp
     */
    private void trainLinearFast(double[][] X, double[] y) {
        int n = X.length;
        int d = X[0].length;
        double[] alpha = new double[n];
        double bias = 0.0;
        int passes = 0;
        Random rng = new Random(42);

        // Pre-calculez K_ii = x_i . x_i (auto-produs scalar) - O(n*d)
        double[] Kii = new double[n];
        for (int i = 0; i < n; i++) Kii[i] = dot(X[i], X[i]);

        // Vectorul w = sum(alpha_i * y_i * x_i) e totul 0 initial
        double[] w = new double[d];

        while (passes < maxPasses) {
            int numChanged = 0;
            for (int i = 0; i < n; i++) {
                // decision(i) = w . X[i] + bias
                double fi = dot(w, X[i]) + bias;
                double Ei = fi - y[i];

                if ((y[i] * Ei < -tol && alpha[i] < C) ||
                    (y[i] * Ei >  tol && alpha[i] > 0)) {

                    int j = i;
                    while (j == i) j = rng.nextInt(n);
                    double fj = dot(w, X[j]) + bias;
                    double Ej = fj - y[j];

                    double aiOld = alpha[i];
                    double ajOld = alpha[j];

                    double L, H;
                    if (y[i] != y[j]) {
                        L = Math.max(0, alpha[j] - alpha[i]);
                        H = Math.min(C, C + alpha[j] - alpha[i]);
                    } else {
                        L = Math.max(0, alpha[i] + alpha[j] - C);
                        H = Math.min(C, alpha[i] + alpha[j]);
                    }
                    if (L == H) continue;

                    double Kij = dot(X[i], X[j]);
                    double eta = 2.0 * Kij - Kii[i] - Kii[j];
                    if (eta >= 0) continue;

                    alpha[j] = ajOld - y[j] * (Ei - Ej) / eta;
                    if (alpha[j] > H) alpha[j] = H;
                    if (alpha[j] < L) alpha[j] = L;
                    if (Math.abs(alpha[j] - ajOld) < 1e-5) continue;

                    alpha[i] = aiOld + y[i] * y[j] * (ajOld - alpha[j]);

                    // Actualizeaza w dupa formula:
                    // w += (alpha_i - aiOld) * y_i * X[i] + (alpha_j - ajOld) * y_j * X[j]
                    double da = (alpha[i] - aiOld) * y[i];
                    double db = (alpha[j] - ajOld) * y[j];
                    for (int k = 0; k < d; k++) {
                        w[k] += da * X[i][k] + db * X[j][k];
                    }

                    double b1 = bias - Ei
                            - y[i] * (alpha[i] - aiOld) * Kii[i]
                            - y[j] * (alpha[j] - ajOld) * Kij;
                    double b2 = bias - Ej
                            - y[i] * (alpha[i] - aiOld) * Kij
                            - y[j] * (alpha[j] - ajOld) * Kii[j];

                    if (alpha[i] > 0 && alpha[i] < C) bias = b1;
                    else if (alpha[j] > 0 && alpha[j] < C) bias = b2;
                    else bias = (b1 + b2) / 2.0;

                    numChanged++;
                }
            }
            if (numChanged == 0) passes++;
            else passes = 0;
        }

        // Salvez vectorii suport (alpha > 0)
        int svCount = 0;
        for (double a : alpha) if (a > 1e-8) svCount++;
        supportX = new double[svCount][];
        supportY = new double[svCount];
        supportA = new double[svCount];
        int k = 0;
        for (int i = 0; i < n; i++) {
            if (alpha[i] > 1e-8) {
                supportX[k] = X[i];
                supportY[k] = y[i];
                supportA[k] = alpha[i];
                k++;
            }
        }
        this.b = bias;
    }

    /** Produs scalar vectorial (x . y). */
    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    public void train(double[][] X, double[] y) {
        if (kernel instanceof LinearKernel) {
            trainLinearFast(X, y);
            return;
        }

        int n = X.length;
        double[] alpha = new double[n];
        double bias = 0.0;
        int passes = 0;
        Random rng = new Random(42);

        // Cache kernel full (poate fi costisitor pentru n mare, dar simplifica mult)
        double[][] K = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double v = kernel.apply(X[i], X[j]);
                K[i][j] = v;
                K[j][i] = v;
            }
        }

        while (passes < maxPasses) {
            int numChanged = 0;
            for (int i = 0; i < n; i++) {
                double Ei = decision(alpha, y, K, i, bias, n) - y[i];
                if ((y[i] * Ei < -tol && alpha[i] < C) ||
                    (y[i] * Ei >  tol && alpha[i] > 0)) {

                    // Alegem al doilea multiplicator la intamplare
                    int j = i;
                    while (j == i) j = rng.nextInt(n);
                    double Ej = decision(alpha, y, K, j, bias, n) - y[j];

                    double aiOld = alpha[i];
                    double ajOld = alpha[j];

                    // Calculeaza L, H
                    double L, H;
                    if (y[i] != y[j]) {
                        L = Math.max(0, alpha[j] - alpha[i]);
                        H = Math.min(C, C + alpha[j] - alpha[i]);
                    } else {
                        L = Math.max(0, alpha[i] + alpha[j] - C);
                        H = Math.min(C, alpha[i] + alpha[j]);
                    }
                    if (L == H) continue;

                    double eta = 2.0 * K[i][j] - K[i][i] - K[j][j];
                    if (eta >= 0) continue;

                    alpha[j] = ajOld - y[j] * (Ei - Ej) / eta;
                    if (alpha[j] > H) alpha[j] = H;
                    if (alpha[j] < L) alpha[j] = L;
                    if (Math.abs(alpha[j] - ajOld) < 1e-5) continue;

                    alpha[i] = aiOld + y[i] * y[j] * (ajOld - alpha[j]);

                    double b1 = bias - Ei
                            - y[i] * (alpha[i] - aiOld) * K[i][i]
                            - y[j] * (alpha[j] - ajOld) * K[i][j];
                    double b2 = bias - Ej
                            - y[i] * (alpha[i] - aiOld) * K[i][j]
                            - y[j] * (alpha[j] - ajOld) * K[j][j];

                    if (alpha[i] > 0 && alpha[i] < C) bias = b1;
                    else if (alpha[j] > 0 && alpha[j] < C) bias = b2;
                    else bias = (b1 + b2) / 2.0;

                    numChanged++;
                }
            }
            if (numChanged == 0) passes++;
            else passes = 0;
        }

        // Pastreaza doar vectorii suport (alpha > 0)
        int svCount = 0;
        for (double a : alpha) if (a > 1e-8) svCount++;
        supportX = new double[svCount][];
        supportY = new double[svCount];
        supportA = new double[svCount];
        int k = 0;
        for (int i = 0; i < n; i++) {
            if (alpha[i] > 1e-8) {
                supportX[k] = X[i];
                supportY[k] = y[i];
                supportA[k] = alpha[i];
                k++;
            }
        }
        this.b = bias;
    }

    /** Functia de decizie peste setul de antrenare (folosita in SMO). */
    private double decision(double[] alpha, double[] y, double[][] K,
                             int idx, double bias, int n) {
        double s = 0;
        for (int i = 0; i < n; i++) s += alpha[i] * y[i] * K[i][idx];
        return s + bias;
    }

    /**
     * @param x vector de trasaturi de clasificat
     * @return valoarea functiei de decizie
     */
    public double decisionFunction(double[] x) {
        // Fast path pentru kernel liniar: f(x) = w.x + b
        // Pentru N=500 si D=8100 reduce 500*8100 operatii la 8100 - ~500x speedup.
        if (kernel instanceof LinearKernel) {
            if (wLinear == null) buildLinearWeight();
            double s = 0;
            for (int i = 0; i < wLinear.length; i++) s += wLinear[i] * x[i];
            return s + b;
        }
        double s = 0;
        for (int i = 0; i < supportX.length; i++) {
            s += supportA[i] * supportY[i] * kernel.apply(supportX[i], x);
        }
        return s + b;
    }

    /** Reconstruieste vectorul w din support vectors pentru kernel liniar. */
    private void buildLinearWeight() {
        if (supportX == null || supportX.length == 0) {
            wLinear = new double[0];
            return;
        }
        int d = supportX[0].length;
        double[] w = new double[d];
        for (int i = 0; i < supportX.length; i++) {
            double c = supportA[i] * supportY[i];
            for (int k = 0; k < d; k++) w[k] += c * supportX[i][k];
        }
        wLinear = w;
    }

    /**
     * @return +1 daca f(x) &gt;= 0, -1 altfel
     */
    public int predict(double[] x) {
        return decisionFunction(x) >= 0 ? 1 : -1;
    }

    /** @return numarul de vectori suport */
    public int numSupportVectors() {
        return supportX == null ? 0 : supportX.length;
    }
}
