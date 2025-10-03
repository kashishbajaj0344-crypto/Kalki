# modules/gui.py
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QPushButton, QTextEdit,
    QLabel, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt

class KalkiGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kalki AI")
        self.setGeometry(200, 200, 800, 600)
        self.init_ui()
        self.show()

    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # PDF ingestion button
        self.ingest_btn = QPushButton("Add PDFs & Ingest")
        self.ingest_btn.clicked.connect(self.add_pdfs)
        layout.addWidget(self.ingest_btn)

        # Chat area
        self.chat_label = QLabel("Ask Kalki anything:")
        layout.addWidget(self.chat_label)

        self.chat_input = QTextEdit()
        self.chat_input.setFixedHeight(100)
        layout.addWidget(self.chat_input)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_query)
        layout.addWidget(self.send_btn)

        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)
        layout.addWidget(self.chat_output)

    def add_pdfs(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDFs to Ingest",
            "",
            "PDF Files (*.pdf);;All Files (*)",
            options=options
        )
        if files:
            QMessageBox.information(self, "PDFs Selected", f"{len(files)} PDFs selected for ingestion.")
            # Call your ingestion function here
            # e.g. from modules.ingest import ingest_files; ingest_files(files)

    def send_query(self):
        query = self.chat_input.toPlainText().strip()
        if not query:
            return
        # Placeholder response; replace with actual RAG query call
        response = f"Kalki would respond to: {query}"
        self.chat_output.append(f"> {query}")
        self.chat_output.append(response)
        self.chat_input.clear()


# Helper function for main.py — ensures QApplication is created first
def start_gui():
    app = QApplication(sys.argv)
    gui = KalkiGUI()
    sys.exit(app.exec())
