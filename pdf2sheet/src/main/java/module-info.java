module com.pdf2sheet {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.apache.pdfbox;
    requires org.apache.poi.ooxml;
    requires java.desktop;

    opens com.pdf2sheet to javafx.fxml;
    exports com.pdf2sheet;
}