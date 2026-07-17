package svm.ui;

import java.awt.*;
import java.awt.event.*;
import java.io.File;
import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

/**
 * Panou pentru cerinta (4): vizualizare imagini salvate si stergere
 * celor necorespunzatoare (neclare, incadrate rau, etc).
 */
public class ViewerPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private final JComboBox<String> personCombo = new JComboBox<>();
    private final DefaultListModel<String> fileModel = new DefaultListModel<>();
    private final JList<String> fileList = new JList<>(fileModel);
    private final JLabel preview = new JLabel("", JLabel.CENTER);
    private final JButton btnRefresh = new JButton("Reincarca");
    private final JButton btnDelete  = new JButton("Sterge selectate");
    private final JButton btnDeletePerson = new JButton("Sterge persoana");
    private final JLabel status = new JLabel(" ");

    public ViewerPanel() {
        super(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JPanel top = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 4));
        top.add(new JLabel("Persoana:"));
        top.add(personCombo);
        top.add(btnRefresh);
        top.add(btnDelete);
        top.add(btnDeletePerson);
        add(top, BorderLayout.NORTH);

        fileList.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
        fileList.setVisibleRowCount(20);
        JScrollPane listScroll = new JScrollPane(fileList);
        listScroll.setPreferredSize(new Dimension(260, 0));

        preview.setBackground(new Color(24, 26, 30));
        preview.setOpaque(true);

        JSplitPane split = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, listScroll, preview);
        split.setDividerLocation(260);
        add(split, BorderLayout.CENTER);
        add(status, BorderLayout.SOUTH);

        personCombo.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) { reloadFiles(); }
        });
        btnRefresh.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) { reloadPersons(); }
        });
        btnDelete.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) { deleteSelected(); }
        });
        btnDeletePerson.addActionListener(new ActionListener() {
            @Override public void actionPerformed(ActionEvent e) { deletePerson(); }
        });
        fileList.addListSelectionListener(new ListSelectionListener() {
            @Override public void valueChanged(ListSelectionEvent e) { showPreview(); }
        });

        reloadPersons();
    }

    private void reloadPersons() {
        personCombo.removeAllItems();
        File[] subs = AppContext.FACES_DIR.listFiles(File::isDirectory);
        if (subs != null) {
            java.util.Arrays.sort(subs);
            for (File f : subs) personCombo.addItem(f.getName());
        }
        reloadFiles();
    }

    private void reloadFiles() {
        fileModel.clear();
        String person = (String) personCombo.getSelectedItem();
        if (person == null) return;
        File dir = new File(AppContext.FACES_DIR, person);
        File[] imgs = dir.listFiles((f, n) -> n.toLowerCase().endsWith(".png")
                                          || n.toLowerCase().endsWith(".jpg"));
        if (imgs != null) {
            java.util.Arrays.sort(imgs);
            for (File f : imgs) fileModel.addElement(f.getName());
        }
        status.setText("Imagini: " + fileModel.size());
    }

    private void showPreview() {
        String name = fileList.getSelectedValue();
        String person = (String) personCombo.getSelectedItem();
        if (name == null || person == null) { preview.setIcon(null); return; }
        File f = new File(new File(AppContext.FACES_DIR, person), name);
        try {
            java.awt.Image im = ImageIO.read(f);
            if (im != null) {
                // scaleaza pentru preview
                java.awt.Image scaled = im.getScaledInstance(384, 384, java.awt.Image.SCALE_SMOOTH);
                preview.setIcon(new ImageIcon(scaled));
                preview.setText("");
            }
        } catch (Exception ex) {
            preview.setIcon(null);
            preview.setText("Eroare: " + ex.getMessage());
        }
    }

    private void deleteSelected() {
        String person = (String) personCombo.getSelectedItem();
        if (person == null) return;
        int[] idx = fileList.getSelectedIndices();
        if (idx.length == 0) return;
        int r = JOptionPane.showConfirmDialog(this,
                "Sterg " + idx.length + " imagini?", "Confirmare",
                JOptionPane.YES_NO_OPTION);
        if (r != JOptionPane.YES_OPTION) return;
        int deleted = 0;
        File dir = new File(AppContext.FACES_DIR, person);
        for (int i = idx.length - 1; i >= 0; i--) {
            String name = fileModel.get(idx[i]);
            if (new File(dir, name).delete()) deleted++;
        }
        status.setText("Sterse: " + deleted);
        reloadFiles();
    }

    /**
     * Sterge complet folderul unei persoane (toate imaginile din faces/&lt;pseudo&gt;/),
     * si - dupa confirmare - si clasificatorul persoanei din classifiers/
     * si vectorii HOG din hog_vectors/.
     */
    private void deletePerson() {
        String person = (String) personCombo.getSelectedItem();
        if (person == null) {
            status.setText("Nicio persoana selectata.");
            return;
        }
        int r = JOptionPane.showConfirmDialog(this,
                "Sterg COMPLET persoana '" + person + "'?\n" +
                "Asta va sterge:\n" +
                "  - toate imaginile din faces/" + person + "/\n" +
                "  - vectorii HOG din hog_vectors/" + person + "/ (daca exista)\n" +
                "  - clasificatorul classifiers/" + person + ".dat (daca exista)",
                "Confirmare stergere persoana", JOptionPane.YES_NO_OPTION);
        if (r != JOptionPane.YES_OPTION) return;

        int deletedFiles = 0;
        deletedFiles += deleteDir(new File(AppContext.FACES_DIR, person));
        deletedFiles += deleteDir(new File(AppContext.HOG_VECTORS_DIR, person));
        File cls = new File(AppContext.CLASSIFIERS_DIR, person + ".dat");
        if (cls.exists() && cls.delete()) deletedFiles++;

        status.setText("Persoana '" + person + "' stearsa. (" + deletedFiles + " fisiere)");
        reloadPersons();
    }

    /** Sterge recursiv un folder. Returneaza numarul de fisiere sterse. */
    private int deleteDir(File dir) {
        if (dir == null || !dir.exists()) return 0;
        int count = 0;
        File[] files = dir.listFiles();
        if (files != null) {
            for (File f : files) {
                if (f.isDirectory()) count += deleteDir(f);
                else if (f.delete()) count++;
            }
        }
        dir.delete();
        return count;
    }
}
