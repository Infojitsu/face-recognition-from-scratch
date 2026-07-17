package svm.ui;

import java.awt.*;
import javax.swing.*;
import svm.io.Storage;
import svm.svm.SVM;

/**
 * Main window. Organizes the application into 4 tabs matching
 * the project requirements:
 *   - Capture (3)
 *   - Viewer (4)
 *   - Training (5, 6, 7 and verifier training for 1)
 *   - Recognition (8)
 */
public class MainFrame extends JFrame {

    private static final long serialVersionUID = 1L;

    public MainFrame() {
        super("Face Recognition - SVM + HOG + SMO");
        AppContext.ensureDirs();

        // Automatically load the head verifier if it exists
        if (AppContext.HEAD_VERIFIER.exists()) {
            try {
                SVM svm = Storage.load(AppContext.HEAD_VERIFIER);
                AppContext.setHeadVerifier(svm);
            } catch (Exception ignored) {}
        }

        JTabbedPane tabs = new JTabbedPane();
        tabs.addTab("1. Capture",     new CapturePanel());
        tabs.addTab("2. Viewer",      new ViewerPanel());
        tabs.addTab("3. Training",    new TrainPanel());
        tabs.addTab("4. Recognition", new RecognizePanel());

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
        JLabel l = new JLabel("Bronto | SVM + HOG + SMO | Sigmoid kernel | Pure Java");
        l.setForeground(new Color(210, 214, 220));
        p.add(l);
        return p;
    }
}
