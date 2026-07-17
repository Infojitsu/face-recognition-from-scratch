package svm.io;

import java.io.*;

/**
 * Serialization / deserialization utilities for classifiers,
 * the SVM head detector and precomputed HOG vectors.
 */
public class Storage {

    /** Serializes any object (implementing Serializable) to a file. */
    public static void save(Object obj, File file) throws IOException {
        file.getParentFile().mkdirs();
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(new FileOutputStream(file)))) {
            oos.writeObject(obj);
        }
    }

    /** Deserializes an object from a file. */
    @SuppressWarnings("unchecked")
    public static <T> T load(File file) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(file)))) {
            return (T) ois.readObject();
        }
    }
}
