package svm.svm;

/**
 * Sigmoid (hyperbolic tangent) kernel: K(x,y) = tanh(alpha * &lt;x,y&gt; + c).
 * It is the kernel required by the problem statement for SMO.
 */
public class SigmoidKernel implements Kernel {

    private static final long serialVersionUID = 1L;

    /** Dot-product scaling */
    private final double alpha;
    /** Offset */
    private final double c;

    /**
     * Constructor.
     * @param alpha scaling factor (typically 1/num_features)
     * @param c offset (typically 0 or a small negative value)
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
