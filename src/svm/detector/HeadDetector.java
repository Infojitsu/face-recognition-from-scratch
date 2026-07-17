package svm.detector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import svm.core.Image;
import svm.core.Rect;
import svm.hog.HOG;
import svm.svm.SVM;

/**
 * Detector de cap bazat pe sliding window + HOG + SVM.
 *
 * Strategia (Dalal &amp; Triggs 2005, clasicul):
 *   1. Construim o piramida de scale ale imaginii (ca sa detectam fete
 *      de diferite dimensiuni cu acelasi clasificator 128x128).
 *   2. Pe fiecare scale, parcurgem imaginea cu o fereastra 128x128 si
 *      pas (stride) de STRIDE pixeli.
 *   3. Pentru fiecare pozitie: extragem HOG-ul, aplicam SVM-ul antrenat
 *      pe "cap vs non-cap". Daca scorul e pozitiv, e candidat.
 *   4. Non-Maximum Suppression: eliminam detectiile suprapuse, pastram
 *      cea cu scor SVM mai mare.
 *
 * Avantaje fata de skin-color:
 *   - Invariant la tonul pielii si iluminare (HOG foloseste gradiente)
 *   - Invariant la fundal - nu mai conteaza draperia, peretele etc.
 *   - Poate detecta mai multe fete simultan
 *
 * Dezavantaj: mai lent. Pe Java pur, aprox 3-5 FPS pentru frame 640x480.
 */
public class HeadDetector {

    /** Scale ale piramidei - acopera de la cap departat (153px) pana la cap
     *  aproape de camera (640px) pe webcam HD. Scale peste min(W,H) sunt
     *  oricum ignorate in detect(), deci nu e cost pentru webcam VGA. */
    private static final double[] SCALES = { 1.2, 1.6, 2.2, 3.0, 4.0, 5.0 };
    /** Pasul (stride) al ferestrei ca fractiune din dimensiunea ferestrei */
    private static final double STRIDE_RATIO = 0.25;
    /** Prag decizie SVM - mai strict, respinge detectiile marginale (gat, frunte partiala etc). */
    private double threshold = 0.25;
    /** Overlap maxim pentru NMS (IoU) - mai mic = suprima mai agresiv patrate suprapuse */
    private static final double NMS_OVERLAP = 0.15;
    /** Prag intersection-over-min-area - suprima cand un patrat mic e continut
     *  in proportie mare in altul (chiar daca nu e perfect). 0.35 suprima cam
     *  orice patrat imbricat in altul mai mare. */
    private static final double NMS_CONTAIN = 0.35;
    /** Dimensiune minima a laturii patratului detectat */
    private static final int MIN_SIDE = 100;
    /** Varianta minima a pixelilor grayscale in fereastra (0-255).
     *  Sub acest prag regiunea e prea uniforma (perete, dulap, ceiling) -
     *  imposibil sa fie cap. Faci brow+skin+eyes genereaza >1000. */
    private static final double MIN_PIXEL_VARIANCE = 400.0;

    /** Clasificatorul SVM cap/non-cap - OBLIGATORIU pentru detectie. */
    private SVM classifier;

    /** Stabilizare temporala pentru stream webcam */
    private Rect lastHead;
    private int framesSinceLastSeen;
    private static final int MAX_COAST_FRAMES = 3;
    private boolean trackingEnabled = false;

    /** Scoruri SVM pentru ultimul apel detect(), in ordinea detectiilor returnate. */
    private List<Double> lastDetectScores = new ArrayList<>();

    public HeadDetector() {}

    public HeadDetector(SVM classifier) {
        this.classifier = classifier;
    }

    /** Seteaza clasificatorul (dupa antrenare sau incarcare). */
    public void setVerifier(SVM c) { this.classifier = c; }

    /** Seteaza pragul SVM (default 0). Mai mic = mai multe detectii. */
    public void setThreshold(double t) { this.threshold = t; }

    /** @return pragul curent de decizie SVM. */
    public double getThreshold() { return threshold; }

    /** @return lista scorurilor SVM pentru ultimul detect(), in aceeasi
     *  ordine cu Rect-urile returnate. Lista e gola daca detect() nu a
     *  returnat nimic. */
    public List<Double> getLastDetectScores() { return lastDetectScores; }

    public void setTrackingEnabled(boolean on) {
        this.trackingEnabled = on;
        if (!on) { lastHead = null; framesSinceLastSeen = 0; }
    }

    public void resetTracking() {
        this.lastHead = null;
        this.framesSinceLastSeen = 0;
    }

    /**
     * Clasa interna pentru o detectie (cu scor SVM).
     */
    private static class Detection {
        Rect rect;
        double score;
        Detection(Rect r, double s) { rect = r; score = s; }
    }

    /**
     * Detecteaza toate capetele din imagine.
     * @return lista de Rect (patrate), sortata dupa scor SVM descrescator
     */
    public List<Rect> detect(Image img) {
        if (classifier == null) {
            // Fara SVM antrenat nu putem detecta.
            return Collections.emptyList();
        }

        List<Detection> all = new ArrayList<>();
        int W = img.getWidth();
        int H = img.getHeight();

        // Parcurgere piramida
        for (double scale : SCALES) {
            int winSize = (int)(HOG.IMG * scale);
            if (winSize > Math.min(W, H)) break;
            if (winSize < MIN_SIDE) continue;

            int step = Math.max(8, (int)(winSize * STRIDE_RATIO));

            for (int y = 0; y + winSize < H; y += step) {
                for (int x = 0; x + winSize < W; x += step) {
                    Image win = img.crop(x, y, x + winSize - 1, y + winSize - 1)
                                   .scale(HOG.IMG, HOG.IMG);
                    double[][] gray = win.toGrayscale();
                    // Filtru rapid: regiuni aproape uniforme (pereti, dulap,
                    // tavan) au varianta mica => sar peste HOG si SVM.
                    if (pixelVariance(gray) < MIN_PIXEL_VARIANCE) continue;
                    double[] feat = HOG.compute(gray);
                    double sc = classifier.decisionFunction(feat);
                    if (sc > threshold) {
                        all.add(new Detection(new Rect(x, y, winSize, winSize), sc));
                    }
                }
            }
        }

        // Non-Maximum Suppression
        List<Detection> kept = nms(all);

        // Sortare dupa scor descrescator
        kept.sort(new Comparator<Detection>() {
            @Override public int compare(Detection a, Detection b) {
                return Double.compare(b.score, a.score);
            }
        });

        List<Rect> out = new ArrayList<>();
        lastDetectScores.clear();
        for (Detection d : kept) {
            out.add(d.rect);
            lastDetectScores.add(d.score);
        }
        return out;
    }

    /**
     * Non-Maximum Suppression: daca doua detectii se suprapun mult,
     * pastram cea cu scor mai mare.
     */
    private List<Detection> nms(List<Detection> dets) {
        List<Detection> sorted = new ArrayList<>(dets);
        sorted.sort(new Comparator<Detection>() {
            @Override public int compare(Detection a, Detection b) {
                return Double.compare(b.score, a.score);
            }
        });
        List<Detection> kept = new ArrayList<>();
        boolean[] suppressed = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (suppressed[i]) continue;
            Detection a = sorted.get(i);
            kept.add(a);
            for (int j = i + 1; j < sorted.size(); j++) {
                if (suppressed[j]) continue;
                Detection b = sorted.get(j);
                // Suprimam daca: (1) IoU ridicat SAU (2) unul e aproape
                // continut in altul (util pentru scale diferite unde IoU
                // e mic dar unul e interior celuilalt).
                if (iou(a.rect, b.rect) > NMS_OVERLAP
                        || containmentRatio(a.rect, b.rect) > NMS_CONTAIN) {
                    suppressed[j] = true;
                }
            }
        }
        return kept;
    }

    /** Varianta pixelilor grayscale - indicator de textura/uniformitate. */
    private static double pixelVariance(double[][] gray) {
        int h = gray.length;
        int w = gray[0].length;
        double sum = 0, sum2 = 0;
        int n = h * w;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double v = gray[y][x];
                sum  += v;
                sum2 += v * v;
            }
        }
        double mean = sum / n;
        return sum2 / n - mean * mean;
    }

    /** Intersection over Union (IoU) pentru doua dreptunghiuri. */
    private static double iou(Rect a, Rect b) {
        int ix1 = Math.max(a.x, b.x);
        int iy1 = Math.max(a.y, b.y);
        int ix2 = Math.min(a.x + a.w, b.x + b.w);
        int iy2 = Math.min(a.y + a.h, b.y + b.h);
        int iw = Math.max(0, ix2 - ix1);
        int ih = Math.max(0, iy2 - iy1);
        int inter = iw * ih;
        int union = a.w * a.h + b.w * b.h - inter;
        return union == 0 ? 0 : (double) inter / union;
    }

    /**
     * Raportul intersectie / min(aria_a, aria_b).
     * Daca un patrat e continut complet in altul, asta returneaza 1.0.
     * Util cand NMS prin IoU nu prinde patrate de scale foarte diferite.
     */
    private static double containmentRatio(Rect a, Rect b) {
        int ix1 = Math.max(a.x, b.x);
        int iy1 = Math.max(a.y, b.y);
        int ix2 = Math.min(a.x + a.w, b.x + b.w);
        int iy2 = Math.min(a.y + a.h, b.y + b.h);
        int iw = Math.max(0, ix2 - ix1);
        int ih = Math.max(0, iy2 - iy1);
        int inter = iw * ih;
        int minArea = Math.min(a.w * a.h, b.w * b.h);
        return minArea == 0 ? 0 : (double) inter / minArea;
    }

    /**
     * Cerinta (2): cap cu scor maxim, scalat la 128x128.
     * NOU: returneaza crop direct din imagine (deja patrat, nu mai are
     * nevoie de scalare in plus daca deja e 128x128).
     */
    public Image largestHead128(Image img) {
        Rect best = largestHeadRect(img);
        if (best == null) return null;
        return img.crop(best.x, best.y, best.x + best.w - 1, best.y + best.h - 1)
                  .scale(HOG.IMG, HOG.IMG);
    }

    /**
     * Returneaza patratul cu scor maxim. Cu tracking pornit: coast pentru
     * cateva frame-uri cand detectorul rateaza momentan.
     */
    public Rect largestHeadRect(Image img) {
        return selectPrimary(detect(img));
    }

    /**
     * Varianta pentru cazul in care apelantul are deja rezultatul detect()
     * - evita apelarea de doua ori a pipeline-ului sliding window.
     * Actualizeaza si starea de tracking (coasting).
     */
    public Rect selectPrimary(List<Rect> heads) {
        if (heads.isEmpty()) {
            if (trackingEnabled && lastHead != null
                    && framesSinceLastSeen < MAX_COAST_FRAMES) {
                framesSinceLastSeen++;
                return lastHead;
            }
            if (framesSinceLastSeen >= MAX_COAST_FRAMES) lastHead = null;
            return null;
        }
        Rect best = heads.get(0);
        framesSinceLastSeen = 0;
        lastHead = best;
        return best;
    }
}
