package svm.util;

import java.io.File;
import java.util.Arrays;
import svm.core.Image;

/**
 * Sursa de imagini dintr-un folder local (fallback cand nu exista webcam).
 * Itereaza prin fisierele .jpg / .png din folder in ordine alfabetica.
 */
public class FolderSource implements ImageSource {

    private final File folder;
    private File[] files;
    private int idx;
    private boolean open;

    public FolderSource(File folder) {
        this.folder = folder;
    }

    @Override
    public void open() throws Exception {
        if (!folder.exists() || !folder.isDirectory()) {
            throw new RuntimeException("Folder inexistent: " + folder);
        }
        files = folder.listFiles(f -> {
            String n = f.getName().toLowerCase();
            return n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png");
        });
        if (files == null) files = new File[0];
        Arrays.sort(files);
        idx = 0;
        open = files.length > 0;
    }

    @Override
    public Image grab() throws Exception {
        if (!open || idx >= files.length) return null;
        Image im = Image.load(files[idx]);
        idx++;
        return im;
    }

    @Override
    public void close() { open = false; }

    @Override
    public boolean isOpen() { return open; }
}
