package svm.detector;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import svm.core.Image;
import svm.hog.HOG;
import svm.io.PersonClassifier;
import svm.io.Storage;
import svm.svm.LinearKernel;
import svm.svm.SVM;
import svm.svm.SigmoidKernel;

/**
 * Training pipeline for:
 *   (a) the "head / non-head" SVM classifier used as a verifier
 *       in HeadDetector. Positive images live in head_images/positive,
 *       negatives in head_images/negative.
 *   (b) one SVM classifier per person (based on the folders
 *       in hog_vectors/&lt;pseudonym&gt;/ which contain the 128x128 images).
 *
 * For (a), a linear kernel is used (fast and sufficient for
 * head/non-head). For (b), the Sigmoid kernel is used, explicitly required
 * by the problem statement for SMO.
 */
public class TrainingPipeline {

    /**
     * Trains the "head / non-head" classifier for sliding-window detection.
     *
     * Positives: the images from positiveDir (scaled to 128x128) + their horizontal flips
     * Negatives: for each image in negativeDir:
     *   - if it is already 128x128, we use it directly
     *   - if it is larger, we extract several random 128x128 patches from it
     *     (this simulates the sliding windows the detector will evaluate
     *     at runtime). This way we train the SVM to reject EXACTLY the kinds of
     *     false-positive windows that the sliding window would generate.
     *
     * @param positiveDir folder with images containing a head (any size)
     * @param negativeDir folder with images without a head (ideally large photos from your room)
     * @return trained SVM
     */
    public static SVM trainHeadVerifier(File positiveDir, File negativeDir) throws Exception {
        List<double[]> feats = new ArrayList<>();
        List<Double> lbl = new ArrayList<>();

        // Positives + horizontal flip (data augmentation)
        int posCount = loadPositivesAugmented(positiveDir, feats, lbl);
        System.out.println("  Positives loaded (with flip): " + posCount);

        // Negatives - the number of patches per image adapts to how many
        // images we have (to keep the total under a reasonable limit).
        int numNegFiles = countImages(negativeDir);
        int negPerImage;
        if (numNegFiles <= 50) negPerImage = 10;     // few photos - many patches
        else if (numNegFiles <= 150) negPerImage = 5;
        else if (numNegFiles <= 400) negPerImage = 3;
        else negPerImage = 2;                         // many photos - few patches
        System.out.println("  Using " + negPerImage + " patches/image "
                + "(for " + numNegFiles + " images)");
        int negCount = loadNegativePatches(negativeDir, feats, lbl, negPerImage);
        System.out.println("  Negatives loaded (patches): " + negCount);

        if (feats.isEmpty() || posCount == 0 || negCount == 0) {
            throw new RuntimeException(
                "Not enough training images found. " +
                "Positives: " + posCount + ", Negatives: " + negCount);
        }

        double[][] X = new double[feats.size()][];
        double[]   y = new double[feats.size()];
        for (int i = 0; i < feats.size(); i++) {
            X[i] = feats.get(i);
            y[i] = lbl.get(i);
        }

        int n = feats.size();
        long kernelMB = (long)n * n * 8 / (1024 * 1024);
        System.out.println("  Training SVM on " + n + " vectors (kernel cache ~"
                + kernelMB + " MB)...");
        System.out.println("  Estimated: "
                + (n < 1500 ? "30s - 2 min" : n < 3000 ? "2-5 min" : "5-15 min"));

        long t0 = System.currentTimeMillis();
        SVM svm = new SVM(new LinearKernel(), 1.0, 1e-3, 10);
        svm.train(X, y);
        long dt = (System.currentTimeMillis() - t0) / 1000;
        System.out.println("  SVM trained in " + dt + "s. SV=" + svm.numSupportVectors());
        return svm;
    }

    /**
     * Loads positives + horizontally flipped versions (doubles the set).
     * @return total number of vectors added
     */
    /** Counts the image files in a folder. */
    private static int countImages(File dir) {
        if (dir == null || !dir.exists()) return 0;
        File[] files = dir.listFiles((f, n) -> {
            String lo = n.toLowerCase();
            return lo.endsWith(".png") || lo.endsWith(".jpg")
                || lo.endsWith(".jpeg") || lo.endsWith(".bmp");
        });
        return files == null ? 0 : files.length;
    }

    private static int loadPositivesAugmented(File dir, List<double[]> feats,
                                                List<Double> lbl) throws Exception {
        if (dir == null || !dir.exists()) return 0;
        File[] files = dir.listFiles((f, n) -> {
            String lo = n.toLowerCase();
            return lo.endsWith(".png") || lo.endsWith(".jpg")
                || lo.endsWith(".jpeg") || lo.endsWith(".bmp");
        });
        if (files == null) return 0;
        int n = 0;
        int skipped = 0;
        for (File f : files) {
            java.awt.image.BufferedImage bi;
            try {
                bi = ImageIO.read(f);
            } catch (Exception ex) {
                System.out.println("  [skip] " + f.getName() + " - read error: " + ex.getMessage());
                skipped++; continue;
            }
            if (bi == null) {
                System.out.println("  [skip] " + f.getName() + " - unknown/unsupported format");
                skipped++; continue;
            }
            Image im = new Image(bi).scale(HOG.IMG, HOG.IMG);
            feats.add(HOG.compute(im));
            lbl.add(1.0);
            n++;
            // Horizontal flip (augmentation)
            Image flipped = flipHorizontal(im);
            feats.add(HOG.compute(flipped));
            lbl.add(1.0);
            n++;
        }
        if (skipped > 0) System.out.println("  Positives skipped: " + skipped);
        return n;
    }

    /**
     * From each large image in dir, extracts random 128x128 patches
     * (at multiple scales) as negatives for the SVM.
     */
    private static int loadNegativePatches(File dir, List<double[]> feats,
                                             List<Double> lbl, int perImage) throws Exception {
        if (dir == null || !dir.exists()) return 0;
        File[] files = dir.listFiles((f, n) -> {
            String lo = n.toLowerCase();
            return lo.endsWith(".png") || lo.endsWith(".jpg")
                || lo.endsWith(".jpeg") || lo.endsWith(".bmp");
        });
        if (files == null) return 0;
        java.util.Random rng = new java.util.Random(42);
        int n = 0;
        int skipped = 0;
        for (File f : files) {
            java.awt.image.BufferedImage bi;
            try {
                bi = ImageIO.read(f);
            } catch (Exception ex) {
                System.out.println("  [skip] " + f.getName() + " - read error: " + ex.getMessage());
                skipped++; continue;
            }
            if (bi == null) {
                System.out.println("  [skip] " + f.getName() + " - unknown/unsupported format");
                skipped++; continue;
            }
            Image im = new Image(bi);
            int W = im.getWidth();
            int H = im.getHeight();

            // If the image is exactly 128x128, use it directly
            if (W <= HOG.IMG + 10 && H <= HOG.IMG + 10) {
                Image scaled = im.scale(HOG.IMG, HOG.IMG);
                feats.add(HOG.compute(scaled));
                lbl.add(-1.0);
                n++;
                continue;
            }

            // Otherwise, extract random patches at variable scales
            for (int k = 0; k < perImage; k++) {
                // Choose a random scale (128 - min(W,H))
                int maxSide = Math.min(W, H);
                int minSide = HOG.IMG;
                int side = minSide + rng.nextInt(Math.max(1, maxSide - minSide));
                if (side > W) side = W;
                if (side > H) side = H;

                int px = rng.nextInt(W - side + 1);
                int py = rng.nextInt(H - side + 1);
                Image patch = im.crop(px, py, px + side - 1, py + side - 1)
                                .scale(HOG.IMG, HOG.IMG);
                feats.add(HOG.compute(patch));
                lbl.add(-1.0);
                n++;
            }
        }
        if (skipped > 0) System.out.println("  Negatives skipped: " + skipped);
        return n;
    }

    /** Horizontal flip of an image. */
    private static Image flipHorizontal(Image img) {
        int W = img.getWidth();
        int H = img.getHeight();
        Image out = new Image(W, H);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                out.setRGB(W - 1 - x, y, img.getR(x, y), img.getG(x, y), img.getB(x, y));
            }
        }
        return out;
    }

    /**
     * Trains one classifier per person
     * (person images = +1, other people's images = -1).
     * @param faceDir folder containing per-person subfolders
     * @param classifierDir folder where the classifiers are saved
     */
    public static void trainPersonClassifiers(File faceDir, File classifierDir) throws Exception {
        File[] subs = faceDir.listFiles(File::isDirectory);
        if (subs == null || subs.length < 2) {
            throw new RuntimeException(
                "At least 2 persons (subfolders) are needed in " + faceDir);
        }

        // Step 1: extract the HOG vectors for each person
        List<String> names = new ArrayList<>();
        List<List<double[]>> perPerson = new ArrayList<>();
        for (File sub : subs) {
            File[] imgs = sub.listFiles((f, n) -> n.toLowerCase().endsWith(".png")
                                              || n.toLowerCase().endsWith(".jpg"));
            if (imgs == null || imgs.length == 0) continue;
            names.add(sub.getName());
            List<double[]> list = new ArrayList<>();
            for (File f : imgs) {
                Image im = new Image(ImageIO.read(f)).scale(HOG.IMG, HOG.IMG);
                list.add(HOG.compute(im));
                // Data augmentation: horizontal flip. Effectively doubles the
                // training set and improves generalization - the human face
                // is reasonably symmetric, the flip does not change identity.
                list.add(HOG.compute(flipHorizontal(im)));
            }
            perPerson.add(list);
        }

        // Step 2: for each person -> one-vs-all SVM
        classifierDir.mkdirs();
        for (int p = 0; p < names.size(); p++) {
            List<double[]> pos = perPerson.get(p);
            List<double[]> neg = new ArrayList<>();
            for (int q = 0; q < names.size(); q++)
                if (q != p) neg.addAll(perPerson.get(q));

            int n = pos.size() + neg.size();
            double[][] X = new double[n][];
            double[]   y = new double[n];
            int k = 0;
            for (double[] v : pos) { X[k] = v; y[k] = +1; k++; }
            for (double[] v : neg) { X[k] = v; y[k] = -1; k++; }

            // Sigmoid kernel (requirement 7). Alpha = 1/100, not 1/d=1/8100.
            // L2-normalized-per-block HOG has dot ~100-150 for similar faces,
            // so alpha*dot ~1.0-1.5 falls in the non-linear region of tanh
            // (tanh(1)=0.76). With alpha=1/d, tanh(0.012)=0.012 - linear regime,
            // final scores ±0.01 = flicker/confusion between persons.
            // maxPasses 20 for more stable SMO convergence with a non-linear
            // kernel (was 10 - sometimes insufficient).
            double alpha = 1.0 / 100.0;
            SVM svm = new SVM(new SigmoidKernel(alpha, 0.0), 1.0, 1e-3, 20);
            svm.train(X, y);

            PersonClassifier pc = new PersonClassifier(names.get(p), svm, X, y);
            Storage.save(pc, new File(classifierDir, names.get(p) + ".dat"));
            System.out.println("  [ok] " + names.get(p)
                    + " | SV=" + svm.numSupportVectors()
                    + " | pos_images=" + pos.size() + " neg_images=" + neg.size());
        }
    }

    /**
     * Loads all person classifiers from the folder.
     */
    public static List<PersonClassifier> loadAll(File classifierDir) throws Exception {
        List<PersonClassifier> out = new ArrayList<>();
        if (!classifierDir.exists()) return out;
        File[] files = classifierDir.listFiles((f, n) -> n.endsWith(".dat"));
        if (files == null) return out;
        for (File f : files) {
            if (f.getName().equals("_head_verifier.dat")) continue;
            out.add(Storage.load(f));
        }
        return out;
    }
}
