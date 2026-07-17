package svm.util;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.lang.reflect.Method;
import svm.core.Image;

/**
 * Sursa de imagini de la camera web, folosind OpenCV Java.
 *
 * IMPORTANT: aceasta clasa foloseste reflexia pentru a se lega de
 * org.opencv.videoio.VideoCapture si org.opencv.core.Mat. Astfel, proiectul
 * poate fi compilat si fara jar-ul OpenCV in classpath. La rulare, daca
 * opencv nu este prezent, metoda open() va arunca o exceptie clara.
 *
 * La rulare, e nevoie de:
 *   -Djava.library.path=lib/opencv
 *   -cp bin;lib/opencv-xxx.jar
 *
 * Cerinta (3) din enunt permite explicit utilizarea OpenCV NUMAI pentru
 * preluarea imaginilor de la camera si desenare. Toti ceilalti algoritmi
 * sunt implementati integral in acest proiect.
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
                    "OpenCV indisponibil. Puneti opencv-xxx.jar in lib/ si " +
                    "folositi -Djava.library.path=. Detaliu: " + t.getMessage());
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

        // Asteapta primul frame valid
        Boolean ok = (Boolean) mRead.invoke(videoCapture, matFrame);
        if (!ok || (Boolean) mEmpty.invoke(matFrame)) {
            throw new RuntimeException("Webcam-ul nu a returnat niciun frame.");
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
        byte[] data = new byte[rows * cols * 3];   // BGR 3 canale
        mGetData.invoke(matFrame, 0, 0, data);

        // Construieste BufferedImage si converteste BGR -> RGB
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
            } catch (Exception e) { /* ignorat */ }
        }
        open = false;
    }

    @Override
    public boolean isOpen() { return open; }
}
