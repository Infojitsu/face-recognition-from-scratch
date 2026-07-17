package svm.util;

import svm.core.Image;

/**
 * Abstractizeaza sursa de imagini (webcam sau folder local).
 * Permite compilare fara OpenCV: cand OpenCV nu exista, programul
 * poate folosi un FolderImageSource pentru dezvoltare/testare.
 */
public interface ImageSource {

    /** Deschide sursa (initializeaza webcam / cursor fisier). */
    void open() throws Exception;

    /** @return urmatoarea imagine sau null daca sursa s-a terminat */
    Image grab() throws Exception;

    /** Elibereaza resursele. */
    void close();

    /** @return true daca sursa mai are imagini */
    boolean isOpen();
}
