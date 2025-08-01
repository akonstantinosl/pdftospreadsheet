package com.pdf2sheet;

import com.benjaminwan.ocrlibrary.OcrEngine;
import com.benjaminwan.ocrlibrary.OcrResult;
import com.benjaminwan.ocrlibrary.TextBlock;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.TextArea;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class PrimaryController {

    @FXML private Button selectPdfButton;
    @FXML private Label selectedFileLabel;
    @FXML private Button processAndSaveButton;
    @FXML private ProgressIndicator progressIndicator;
    @FXML private TextArea logArea;

    private File selectedPdfFile;
    private OcrEngine ocrEngine;

    @FXML
    public void initialize() {
        // Inisialisasi OCR Engine di sini
        // Pastikan folder 'models' dan 'jni-libs' ada di direktori yang sama dengan JAR Anda
        log("Menginisialisasi OCR Engine...");
        try {
            ocr_engine_holder.engine = new OcrEngine();
            log("Versi JNI: " + ocr_engine_holder.engine.getVersion());

            File modelsDir = new File("models");
            if (!modelsDir.exists() || !modelsDir.isDirectory()) {
                log("ERROR: Folder 'models' tidak ditemukan!");
                return;
            }

            // Inisialisasi model OCR
            boolean modelsInitialized = ocr_engine_holder.engine.initModels(
                    modelsDir.getAbsolutePath(),
                    "ch_PP-OCRv3_det_infer.onnx",
                    "ch_ppocr_mobile_v2.0_cls_infer.onnx",
                    "ch_PP-OCRv3_rec_infer.onnx",
                    "ppocr_keys_v1.txt"
            );

            if (modelsInitialized) {
                log("OCR Engine berhasil diinisialisasi.");
            } else {
                log("ERROR: Gagal memuat model OCR. Pastikan semua file model ada di folder 'models'.");
            }
        } catch (Exception e) {
            log("FATAL: Gagal memuat library JNI. Pastikan folder 'jni-libs' ada dan berisi file yang benar untuk OS Anda.");
            log("Error: " + e.getMessage());
        }
    }

    // Menggunakan kelas statis untuk memastikan hanya ada satu instance OcrEngine
    private static class ocr_engine_holder {
        static OcrEngine engine;
    }

    @FXML
    private void handleSelectPdf() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Pilih File PDF");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("PDF Files", "*.pdf"));
        Stage stage = (Stage) selectPdfButton.getScene().getWindow();
        selectedPdfFile = fileChooser.showOpenDialog(stage);

        if (selectedPdfFile != null) {
            selectedFileLabel.setText(selectedPdfFile.getName());
            processAndSaveButton.setDisable(false);
            log("File dipilih: " + selectedPdfFile.getAbsolutePath());
        }
    }

    @FXML
    private void handleProcessAndSave() {
        if (selectedPdfFile == null) return;
        if (ocr_engine_holder.engine == null) {
            log("ERROR: OCR Engine tidak siap.");
            return;
        }

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Simpan File Excel");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Excel Workbook", "*.xlsx"));
        fileChooser.setInitialFileName(selectedPdfFile.getName().replace(".pdf", ".xlsx"));
        Stage stage = (Stage) processAndSaveButton.getScene().getWindow();
        File saveFile = fileChooser.showSaveDialog(stage);

        if (saveFile != null) {
            // Jalankan proses di background thread agar UI tidak freeze
            Task<Void> processingTask = new Task<>() {
                @Override
                protected Void call() throws Exception {
                    Platform.runLater(() -> {
                        setControlsDisabled(true);
                        progressIndicator.setVisible(true);
                        logArea.clear();
                        log("Memulai proses konversi...");
                    });

                    try {
                        // 1. Baca PDF dan konversi ke gambar
                        List<List<List<String>>> allPagesData = processPdfToTextData(selectedPdfFile);

                        // 2. Tulis data ke Excel
                        log("Menyimpan data ke file Excel...");
                        saveDataToExcel(allPagesData, saveFile);
                        Platform.runLater(() -> log("Sukses! File disimpan di: " + saveFile.getAbsolutePath()));

                    } catch (Exception e) {
                        Platform.runLater(() -> log("ERROR: Terjadi kesalahan saat proses. Lihat console untuk detail."));
                        e.printStackTrace();
                    } finally {
                        Platform.runLater(() -> {
                            setControlsDisabled(false);
                            progressIndicator.setVisible(false);
                        });
                    }
                    return null;
                }
            };
            new Thread(processingTask).start();
        }
    }

    private List<List<List<String>>> processPdfToTextData(File pdfFile) throws IOException {
        List<List<List<String>>> allPagesData = new ArrayList<>();
        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer pdfRenderer = new PDFRenderer(document);
            int numPages = document.getNumberOfPages();
            Platform.runLater(() -> log(String.format("Mendeteksi %d halaman di PDF.", numPages)));

            for (int i = 0; i < numPages; i++) {
                final int pageNum = i + 1;
                Platform.runLater(() -> log(String.format("Memproses Halaman %d dari %d...", pageNum, numPages)));

                // Render halaman PDF ke gambar
                BufferedImage bim = pdfRenderer.renderImageWithDPI(i, 300); // DPI tinggi untuk akurasi
                File tempImageFile = File.createTempFile("pdf-page-" + pageNum, ".png");
                ImageIO.write(bim, "png", tempImageFile);

                // Jalankan OCR pada gambar
                Platform.runLater(() -> log("  -> Menjalankan OCR pada Halaman " + pageNum));
                OcrResult ocrResult = ocr_engine_holder.engine.detect(tempImageFile.getAbsolutePath(), 1024);
                tempImageFile.delete(); // Hapus file sementara

                if (ocrResult == null || ocrResult.getTextBlocks().isEmpty()) {
                    Platform.runLater(() -> log("  -> Tidak ada teks yang terdeteksi di Halaman " + pageNum));
                    continue;
                }

                // Susun blok teks menjadi tabel
                List<List<String>> pageData = structureTextBlocks(ocrResult.getTextBlocks());
                allPagesData.add(pageData);
                Platform.runLater(() -> log(String.format("  -> Halaman %d selesai diproses. Ditemukan %d baris data.", pageNum, pageData.size())));
            }
        }
        return allPagesData;
    }

    private List<List<String>> structureTextBlocks(ArrayList<TextBlock> textBlocks) {
        if (textBlocks.isEmpty()) return new ArrayList<>();

        // 1. Kelompokkan text block menjadi baris berdasarkan posisi Y
        textBlocks.sort(Comparator.comparingInt(tb -> tb.getBoxPoint().get(0).getY()));

        List<List<TextBlock>> rows = new ArrayList<>();
        List<TextBlock> currentRow = new ArrayList<>();
        currentRow.add(textBlocks.get(0));

        for (int i = 1; i < textBlocks.size(); i++) {
            TextBlock prev = currentRow.get(currentRow.size() - 1);
            TextBlock current = textBlocks.get(i);

            // Cek apakah pusat Y dari blok saat ini berada dalam rentang tinggi blok sebelumnya
            int prevCenterY = (prev.getBoxPoint().get(0).getY() + prev.getBoxPoint().get(3).getY()) / 2;
            int currentCenterY = (current.getBoxPoint().get(0).getY() + current.getBoxPoint().get(3).getY()) / 2;
            int prevHeight = prev.getBoxPoint().get(3).getY() - prev.getBoxPoint().get(0).getY();

            if (Math.abs(currentCenterY - prevCenterY) < prevHeight * 0.7) {
                currentRow.add(current);
            } else {
                rows.add(currentRow);
                currentRow = new ArrayList<>();
                currentRow.add(current);
            }
        }
        rows.add(currentRow);

        // 2. Di setiap baris, urutkan berdasarkan posisi X
        for (List<TextBlock> row : rows) {
            row.sort(Comparator.comparingInt(tb -> tb.getBoxPoint().get(0).getX()));
        }

        // 3. Konversi ke List<List<String>>
        return rows.stream()
                .map(row -> row.stream()
                        .map(TextBlock::getText)
                        .collect(Collectors.toList()))
                .collect(Collectors.toList());
    }

    private void saveDataToExcel(List<List<List<String>>> allPagesData, File file) throws IOException {
        try (Workbook workbook = new XSSFWorkbook()) {
            for (int i = 0; i < allPagesData.size(); i++) {
                Sheet sheet = workbook.createSheet("Halaman " + (i + 1));
                List<List<String>> pageData = allPagesData.get(i);
                int maxCols = pageData.stream().mapToInt(List::size).max().orElse(0);

                for (int r = 0; r < pageData.size(); r++) {
                    Row excelRow = sheet.createRow(r);
                    List<String> rowData = pageData.get(r);
                    for (int c = 0; c < rowData.size(); c++) {
                        Cell cell = excelRow.createCell(c);
                        cell.setCellValue(rowData.get(c));
                    }
                }
                // Auto-size columns for better readability
                for(int j=0; j<maxCols; j++){
                    sheet.autoSizeColumn(j);
                }
            }

            try (FileOutputStream fileOut = new FileOutputStream(file)) {
                workbook.write(fileOut);
            }
        }
    }

    private void setControlsDisabled(boolean disabled) {
        selectPdfButton.setDisable(disabled);
        processAndSaveButton.setDisable(disabled);
    }

    private void log(String message) {
        Platform.runLater(() -> logArea.appendText(message + "\n"));
    }
}