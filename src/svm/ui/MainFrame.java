package svm.ui;

import java.awt.*;
import javax.swing.*;
import svm.io.Storage;
import svm.svm.SVM;

/**
 * Fereastra principala. Organizeaza aplicatia in 4 tab-uri corespunzatoare
 * cerintelor proiectului:
 *   - Capturare (3)
 *   - Vizualizare (4)
 *   - Antrenare (5, 6, 7 si antrenare verificator pentru 1)
 *   - Recunoastere (8)
 */
public class MainFrame extends JFrame {

    private static final long serialVersionUID = 1L;

    public MainFrame() {
        super("Recunoastere Fete - SVM + HOG + SMO");
        AppContext.ensureDirs();

        // Incarca automat verificatorul cap daca exista
        if (AppContext.HEAD_VERIFIER.exists()) {
            try {
                SVM svm = Storage.load(AppContext.HEAD_VERIFIER);
                AppContext.setHeadVerifier(svm);
            } catch (Exception ignored) {}
        }

        JTabbedPane tabs = new JTabbedPane();
        tabs.addTab("1. Capturare",    new CapturePanel());
        tabs.addTab("2. Vizualizare",  new ViewerPanel());
        tabs.addTab("3. Antrenare",    new TrainPanel());
        tabs.addTab("4. Recunoastere", new RecognizePanel());

        setLayout(new BorderLayout());
        add(tabs, BorderLayout.CENTER);
        add(footer(), BorderLayout.SOUTH);

        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(1000, 700);
        setLocationRelativeTo(null);
    }

    private JComponent footer() {
        JPanel p = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 4));
        p.setBackground(new Color(36, 38, 44));
        JLabel l = new JLabel("Bronto | SVM + HOG + SMO | Sigmoid kernel | Java pur");
        l.setForeground(new Color(210, 214, 220));
        p.add(l);
        return p;
    }
}
