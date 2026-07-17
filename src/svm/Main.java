package svm;

import javax.swing.SwingUtilities;
import javax.swing.UIManager;
import svm.ui.MainFrame;

/**
 * Application entry point.
 * Run:  java -cp bin svm.Main
 * Or: java -cp "bin;lib/opencv-xxx.jar" -Djava.library.path=lib svm.Main
 */
public class Main {

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {}
        SwingUtilities.invokeLater(new Runnable() {
            @Override public void run() {
                new MainFrame().setVisible(true);
            }
        });
    }
}
