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
 * Pipeline de antrenare pentru:
 *   (a) clasificatorul SVM "cap / nu cap" folosit ca verificator
 *       in HeadDetector. Imaginile pozitive sunt in head_images/positive,
 *       negativele in head_images/negative.
 *   (b) un clasificator SVM pentru fiecare persoana (pe baza folderelor
 *       din hog_vectors/&lt;pseudonym&gt;/ care contin imaginile 128x128).
 *
 * Pentru (a), se foloseste kernel liniar (rapid si suficient pentru
 * cap/non-cap). Pentru (b), se foloseste kernelul Sigmoid, cerut explicit
 * in enunt pentru SMO.
 */
public class TrainingPipeline {

    /**
     * Antreneaza clasificatorul "cap / non-cap" pentru detectie sliding window.
     *
     * Poziti:  imaginile din positiveDir (scalate 128x128) + flipurile lor orizontale
     * Negative: pentru fiecare imagine din negativeDir:
     *   - daca e deja 128x128, o folosim direct
     *   - daca e mai mare, extragem mai multe patch-uri 128x128 aleatoare din ea
     *     (asta simuleaza ferestrele sliding pe care detectorul le va evalua
     *     la runtime). Asa trainam SVM-ul sa respinga EXACT tipurile de
     *     ferestre false positives pe care sliding window le-ar genera.
     *
     * @param positiveDir folder cu imagini care contin cap (orice dimensiune)
     * @param negativeDir folder cu imagini fara cap (ideal poze mari din camera ta)
     * @return SVM antrenat
     */
    public static SVM trainHeadVerifier(File positiveDir, File negativeDir) throws Exception {
        List<double[]> feats = new ArrayList<>();
        List<Double> lbl = new ArrayList<>();

        // Pozitive + flip orizontal (data augmentation)
        int posCount = loadPositivesAugmented(positiveDir, feats, lbl);
        System.out.println("  Pozitive incarcate (cu flip): " + posCount);

        // Negative - numar de patch-uri per imagine se adapteaza la cate
        // imagini avem (pentru a tine totalul sub o limita rezonabila).
        int numNegFiles = countImages(negativeDir);
        int negPerImage;
        if (numNegFiles <= 50) negPerImage = 10;     // putine poze - multe patch-uri
        else if (numNegFiles <= 150) negPerImage = 5;
        else if (numNegFiles <= 400) negPerImage = 3;
        else negPerImage = 2;                         // multe poze - putine patch-uri
        System.out.println("  Folosesc " + negPerImage + " patch-uri/imagine "
                + "(pentru " + numNegFiles + " imagini)");
        int negCount = loadNegativePatches(negativeDir, feats, lbl, negPerImage);
        System.out.println("  Negative incarcate (patch-uri): " + negCount);

        if (feats.isEmpty() || posCount == 0 || negCount == 0) {
            throw new RuntimeException(
                "Nu s-au gasit imagini de antrenare suficiente. " +
                "Pozitive: " + posCount + ", Negative: " + negCount);
        }

        double[][] X = new double[feats.size()][];
        double[]   y = new double[feats.size()];
        for (int i = 0; i < feats.size(); i++) {
            X[i] = feats.get(i);
            y[i] = lbl.get(i);
        }

        int n = feats.size();
        long kernelMB = (long)n * n * 8 / (1024 * 1024);
        System.out.println("  Antrenez SVM pe " + n + " vectori (cache kernel ~"
                + kernelMB + " MB)...");
        System.out.println("  Estimat: "
                + (n < 1500 ? "30s - 2 min" : n < 3000 ? "2-5 min" : "5-15 min"));

        long t0 = System.currentTimeMillis();
        SVM svm = new SVM(new LinearKernel(), 1.0, 1e-3, 10);
        svm.train(X, y);
        long dt = (System.currentTimeMillis() - t0) / 1000;
        System.out.println("  SVM antrenat in " + dt + "s. SV=" + svm.numSupportVectors());
        return svm;
    }

    /**
     * Incarca pozitive + versiuni flip orizontal (dublezi setul).
     * @return numar total de vectori adaugati
     */
    /** Numara fisierele imagine dintr-un folder. */
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
                System.out.println("  [skip] " + f.getName() + " - eroare citire: " + ex.getMessage());
                skipped++; continue;
            }
            if (bi == null) {
                System.out.println("  [skip] " + f.getName() + " - format necunoscut/nesuportat");
                skipped++; continue;
            }
            Image im = new Image(bi).scale(HOG.IMG, HOG.IMG);
            feats.add(HOG.compute(im));
            lbl.add(1.0);
            n++;
            // Flip orizontal (augmentare)
            Image flipped = flipHorizontal(im);
            feats.add(HOG.compute(flipped));
            lbl.add(1.0);
            n++;
        }
        if (skipped > 0) System.out.println("  Pozitive skipped: " + skipped);
        return n;
    }

    /**
     * Din fiecare imagine mare din dir, extrage patch-uri 128x128 aleatoare
     * (la multiple scale) ca negative pentru SVM.
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
                System.out.println("  [skip] " + f.getName() + " - eroare citire: " + ex.getMessage());
                skipped++; continue;
            }
            if (bi == null) {
                System.out.println("  [skip] " + f.getName() + " - format necunoscut/nesuportat");
                skipped++; continue;
            }
            Image im = new Image(bi);
            int W = im.getWidth();
            int H = im.getHeight();

            // Daca imaginea e chiar 128x128, o luam direct
            if (W <= HOG.IMG + 10 && H <= HOG.IMG + 10) {
                Image scaled = im.scale(HOG.IMG, HOG.IMG);
                feats.add(HOG.compute(scaled));
                lbl.add(-1.0);
                n++;
                continue;
            }

            // Altfel, extragem patch-uri aleatoare la scale variabile
            for (int k = 0; k < perImage; k++) {
                // Alege o scala aleatoare (128 - min(W,H))
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
        if (skipped > 0) System.out.println("  Negative skipped: " + skipped);
        return n;
    }

    /** Flip orizontal al unei imagini. */
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
     * Antreneaza un clasificator pentru fiecare persoana
     * (imagini persoana = +1, imagini altor persoane = -1).
     * @param faceDir folder care contine subfoldere per persoana
     * @param classifierDir folder unde se salveaza clasificatoarele
     */
    public static void trainPersonClassifiers(File faceDir, File classifierDir) throws Exception {
        File[] subs = faceDir.listFiles(File::isDirectory);
        if (subs == null || subs.length < 2) {
            throw new RuntimeException(
                "E nevoie de cel putin 2 persoane (subfoldere) in " + faceDir);
        }

        // Pasul 1: extrage vectorii HOG pentru fiecare persoana
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
                // Data augmentation: flip orizontal. Dubleaza efectiv setul
                // de antrenare si imbunatateste generalizarea - fata umana
                // e rezonabil de simetrica, flip-ul nu schimba identitatea.
                list.add(HOG.compute(flipHorizontal(im)));
            }
            perPerson.add(list);
        }

        // Pasul 2: pentru fiecare persoana -> SVM one-vs-all
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

            // Kernel Sigmoid (cerinta 7). Alpha = 1/100, nu 1/d=1/8100.
            // HOG L2-normalized-per-block are dot ~100-150 pentru fete similare,
            // deci alpha*dot ~1.0-1.5 cade in regiunea neliniara a tanh
            // (tanh(1)=0.76). Cu alpha=1/d, tanh(0.012)=0.012 - regim liniar,
            // scoruri finale ±0.01 = flicker/confuzie intre persoane.
            // maxPasses 20 pentru convergenta SMO mai stabila la kernel
            // non-liniar (era 10 - uneori insuficient).
            double alpha = 1.0 / 100.0;
            SVM svm = new SVM(new SigmoidKernel(alpha, 0.0), 1.0, 1e-3, 20);
            svm.train(X, y);

            PersonClassifier pc = new PersonClassifier(names.get(p), svm, X, y);
            Storage.save(pc, new File(classifierDir, names.get(p) + ".dat"));
            System.out.println("  [ok] " + names.get(p)
                    + " | SV=" + svm.numSupportVectors()
                    + " | imagini_pos=" + pos.size() + " imagini_neg=" + neg.size());
        }
    }

    /**
     * Incarca toate clasificatoarele de persoana din folder.
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
