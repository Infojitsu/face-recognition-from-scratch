package svm.util;

import svm.core.Image;

/**
 * Abstracts the image source (webcam or local folder).
 * Allows compilation without OpenCV: when OpenCV is not present, the program
 * can use a FolderImageSource for development/testing.
 */
public interface ImageSource {

    /** Opens the source (initializes the webcam / file cursor). */
    void open() throws Exception;

    /** @return the next image or null if the source is exhausted */
    Image grab() throws Exception;

    /** Releases the resources. */
    void close();

    /** @return true if the source still has images */
    boolean isOpen();
}
