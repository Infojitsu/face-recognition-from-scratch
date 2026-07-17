package svm.svm;

/**
 * Kernel Sigmoid (hiperbolic tangent): K(x,y) = tanh(alpha * &lt;x,y&gt; + c).
 * Este kernelul cerut in enuntul problemei pentru SMO.
 */
public class SigmoidKernel implements Kernel {

    private static final long serialVersionUID = 1L;

    /** Scalare produs scalar */
    private final double alpha;
    /** Deplasare */
    private final double c;

    /**
     * Constructor.
     * @param alpha factor de scalare (tipic 1/nr_caracteristici)
     * @param c deplasare (tipic 0 sau mic negativ)
     */
    public SigmoidKernel(double alpha, double c) {
        this.alpha = alpha;
        this.c = c;
    }

    @Override
    public double apply(double[] x, double[] y) {
        double dot = 0;
        for (int i = 0; i < x.length; i++) dot += x[i] * y[i];
        return Math.tanh(alpha * dot + c);
    }
}
