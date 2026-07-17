package svm.svm;

import java.io.Serializable;
import java.util.Random;

/**
 * Binary SVM classifier (+1/-1) trained with the SMO algorithm
 * (Sequential Minimal Optimization), Platt 1998.
 *
 * Solves the dual problem:
 *   max    sum_i alpha_i - 0.5 * sum_ij alpha_i alpha_j y_i y_j K(x_i, x_j)
 *   s.t.   0 &lt;= alpha_i &lt;= C
 *          sum_i alpha_i y_i = 0
 *
 * Decision: f(x) = sum_i alpha_i y_i K(x_i, x) - b
 * Class: sign(f(x))
 *
 * When saving via serialization, we keep only the support vectors (alpha_i > 0)
 * to minimize file size.
 */
public class SVM implements Serializable {

    private static final long serialVersionUID = 1L;

    /** The kernel used (Sigmoid by default) */
    private Kernel kernel;
    /** The support vectors (after training) */
    private double[][] supportX;
    /** The support vector labels */
    private double[] supportY;
    /** The Lagrange multipliers for the support vectors */
    private double[] supportA;
    /** The bias term (b) */
    private double b;
    /** The regularization parameter C */
    private double C;
    /** Convergence tolerance */
    private double tol;
    /** Maximum number of iterations without changes */
    private int maxPasses;
    /** Cache for the linear kernel: w = sum(alpha_i * y_i * x_i).
     *  Transient - rebuilt on demand after deserialization.
     *  Collapses f(x) from O(N*D) (over N support vectors) to O(D) (a single dot).
     */
    private transient double[] wLinear;

    /**
     * Constructor with kernel and SMO parameters.
     * @param kernel the kernel function
     * @param C regularization (typically 1.0)
     * @param tol tolerance (typically 1e-3)
     * @param maxPasses iterations without changes until stopping (typically 10)
     */
    public SVM(Kernel kernel, double C, double tol, int maxPasses) {
        this.kernel = kernel;
        this.C = C;
        this.tol = tol;
        this.maxPasses = maxPasses;
    }

    /**
     * Trains the classifier on the set (X, y) with y in {-1, +1}.
     * Implements Platt's simplified SMO algorithm.
     */
    /**
     * For the linear kernel, no kernel matrix is needed (O(n^2) memory)
     * - we can keep the vector w = sum(alpha_i * y_i * x_i) directly in O(n*d).
     * K(x_i, x_j) = x_i . x_j is computed on the fly in O(d).
     *
     * Result: dramatically lower memory and time for large n.
     * E.g.: n=3000, d=8100:
     *   - full cache: 72 MB memory, ~30 min time
     *   - fast linear: ~200 KB extra memory, ~2 min time
     */
    private void trainLinearFast(double[][] X, double[] y) {
        int n = X.length;
        int d = X[0].length;
        double[] alpha = new double[n];
        double bias = 0.0;
        int passes = 0;
        Random rng = new Random(42);

        // Precompute K_ii = x_i . x_i (self dot product) - O(n*d)
        double[] Kii = new double[n];
        for (int i = 0; i < n; i++) Kii[i] = dot(X[i], X[i]);

        // The vector w = sum(alpha_i * y_i * x_i) is all zeros initially
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

                    // Update w using the formula:
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

        // Save the support vectors (alpha > 0)
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

    /** Vector dot product (x . y). */
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

        // Full kernel cache (can be expensive for large n, but simplifies a lot)
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

                    // Choose the second multiplier at random
                    int j = i;
                    while (j == i) j = rng.nextInt(n);
                    double Ej = decision(alpha, y, K, j, bias, n) - y[j];

                    double aiOld = alpha[i];
                    double ajOld = alpha[j];

                    // Compute L, H
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

        // Keep only the support vectors (alpha > 0)
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

    /** The decision function over the training set (used inside SMO). */
    private double decision(double[] alpha, double[] y, double[][] K,
                             int idx, double bias, int n) {
        double s = 0;
        for (int i = 0; i < n; i++) s += alpha[i] * y[i] * K[i][idx];
        return s + bias;
    }

    /**
     * @param x feature vector to classify
     * @return the value of the decision function
     */
    public double decisionFunction(double[] x) {
        // Fast path for the linear kernel: f(x) = w.x + b
        // For N=500 and D=8100 it reduces 500*8100 operations to 8100 - ~500x speedup.
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

    /** Rebuilds the w vector from support vectors for the linear kernel. */
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
     * @return +1 if f(x) &gt;= 0, -1 otherwise
     */
    public int predict(double[] x) {
        return decisionFunction(x) >= 0 ? 1 : -1;
    }

    /** @return the number of support vectors */
    public int numSupportVectors() {
        return supportX == null ? 0 : supportX.length;
    }
}
