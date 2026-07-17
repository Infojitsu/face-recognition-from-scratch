package svm.svm;

/**
 * Kernel liniar K(x,y) = &lt;x,y&gt;. Utilizat optional pentru antrenarea
 * detectorului de cap (mai rapid decat Sigmoid).
 */
public class LinearKernel implements Kernel {

    private static final long serialVersionUID = 1L;

    @Override
    public double apply(double[] x, double[] y) {
        double s = 0;
        for (int i = 0; i < x.length; i++) s += x[i] * y[i];
        return s;
    }
}
