package svm.io;

import java.io.*;

/**
 * Utilitare de serializare / deserializare pentru clasificatoare,
 * detector-cap SVM si vectori HOG precalculati.
 */
public class Storage {

    /** Serializeaza orice obiect (care implementeaza Serializable) la fisier. */
    public static void save(Object obj, File file) throws IOException {
        file.getParentFile().mkdirs();
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(new FileOutputStream(file)))) {
            oos.writeObject(obj);
        }
    }

    /** Deserializeaza un obiect dintr-un fisier. */
    @SuppressWarnings("unchecked")
    public static <T> T load(File file) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(file)))) {
            return (T) ois.readObject();
        }
    }
}
