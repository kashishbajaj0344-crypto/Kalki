"""
gui.py
Simple GUI for Kalki Phase 1 using Tkinter.
"""

import tkinter as tk
from tkinter import scrolledtext
from modules.llm import LLMEngine
from modules.ingest import DocumentIngestor
import logging

logger = logging.getLogger(__name__)

class KalkiGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Kalki Phase 1")
        self.root.geometry("800x600")

        # LLM Engine
        self.llm = LLMEngine()
        self.ingestor = DocumentIngestor()

        # Query input
        self.input_label = tk.Label(self.root, text="Enter query:")
        self.input_label.pack()
        self.input_text = tk.Entry(self.root, width=100)
        self.input_text.pack()

        # Output display
        self.output_text = scrolledtext.ScrolledText(self.root, width=100, height=25)
        self.output_text.pack()

        # Buttons
        self.query_button = tk.Button(self.root, text="Run RAG Query", command=self.run_query)
        self.query_button.pack(pady=5)

        self.ingest_button = tk.Button(self.root, text="Ingest PDFs", command=self.run_ingest)
        self.ingest_button.pack(pady=5)

    def run_query(self):
        query = self.input_text.get()
        if query.strip():
            response = self.llm.rag_query(query)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, response)

    def run_ingest(self):
        self.ingestor.ingest_all()
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "PDF ingestion completed.")

    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = KalkiGUI()
    gui.start()
