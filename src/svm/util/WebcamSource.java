package svm.util;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.lang.reflect.Method;
import svm.core.Image;

/**
 * Image source from the webcam, using OpenCV Java.
 *
 * IMPORTANT: this class uses reflection to bind to
 * org.opencv.videoio.VideoCapture and org.opencv.core.Mat. This way, the project
 * can be compiled even without the OpenCV jar on the classpath. At runtime, if
 * opencv is not present, the open() method throws a clear exception.
 *
 * At runtime, you need:
 *   -Djava.library.path=lib/opencv
 *   -cp bin;lib/opencv-xxx.jar
 *
 * Requirement (3) of the problem statement explicitly allows using OpenCV ONLY for
 * grabbing images from the camera and drawing. All other algorithms
 * are implemented entirely in this project.
 */
public class WebcamSource implements ImageSource {

    private Object videoCapture;  // org.opencv.videoio.VideoCapture
    private Object matFrame;      // org.opencv.core.Mat
    private boolean open = false;

    private Method mRead;
    private Method mEmpty;
    private Method mGetRows;
    private Method mGetCols;
    private Method mGetData;

    @Override
    public void open() throws Exception {
        try {
            Class<?> coreClass = Class.forName("org.opencv.core.Core");
            String nativeLib = (String) coreClass.getField("NATIVE_LIBRARY_NAME").get(null);
            System.loadLibrary(nativeLib);
        } catch (Throwable t) {
            throw new RuntimeException(
                    "OpenCV unavailable. Put opencv-xxx.jar in lib/ and " +
                    "use -Djava.library.path=. Detail: " + t.getMessage());
        }
        Class<?> vcClass = Class.forName("org.opencv.videoio.VideoCapture");
        videoCapture = vcClass.getConstructor(int.class).newInstance(0);
        Class<?> matClass = Class.forName("org.opencv.core.Mat");
        matFrame = matClass.getConstructor().newInstance();
        mRead  = vcClass.getMethod("read", matClass);
        mEmpty = matClass.getMethod("empty");
        mGetRows = matClass.getMethod("rows");
        mGetCols = matClass.getMethod("cols");
        mGetData = matClass.getMethod("get", int.class, int.class, byte[].class);

        // Wait for the first valid frame
        Boolean ok = (Boolean) mRead.invoke(videoCapture, matFrame);
        if (!ok || (Boolean) mEmpty.invoke(matFrame)) {
            throw new RuntimeException("The webcam did not return any frame.");
        }
        open = true;
    }

    @Override
    public Image grab() throws Exception {
        if (!open) return null;
        Boolean ok = (Boolean) mRead.invoke(videoCapture, matFrame);
        if (!ok || (Boolean) mEmpty.invoke(matFrame)) return null;
        int rows = (int) mGetRows.invoke(matFrame);
        int cols = (int) mGetCols.invoke(matFrame);
        byte[] data = new byte[rows * cols * 3];   // BGR 3 channels
        mGetData.invoke(matFrame, 0, 0, data);

        // Build a BufferedImage and convert BGR -> RGB
        BufferedImage bi = new BufferedImage(cols, rows, BufferedImage.TYPE_3BYTE_BGR);
        byte[] dest = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
        System.arraycopy(data, 0, dest, 0, data.length);
        return new Image(bi);
    }

    @Override
    public void close() {
        if (videoCapture != null) {
            try {
                videoCapture.getClass().getMethod("release").invoke(videoCapture);
            } catch (Exception e) { /* ignored */ }
        }
        open = false;
    }

    @Override
    public boolean isOpen() { return open; }
}
