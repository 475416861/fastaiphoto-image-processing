"""
image_processing_gui.py
======================

This module implements a simple graphical user interface (GUI) for the
document image processing pipeline defined in ``image_processing.py``.
The GUI is built with PyQt5 and allows the user to select input and
output directories, choose which processing steps to apply (deskew,
crop, border removal, denoising) via check boxes, and run the batch
process.  A log area shows progress messages as files are processed.

Because PyQt5 is not installed in the execution environment, this
script cannot be run here, but it serves as a reference implementation
that you can execute on a machine with PyQt5 available.  To install
PyQt5 locally, run::

    pip install pyqt5

Then run the GUI with::

    python image_processing_gui.py

"""

import os
import threading
from functools import partial

"""
Try to import Qt classes from PyQt5.  If PyQt5 is not available,
fall back to PySide2.  Both bindings provide nearly identical APIs
for the widgets used in this script.  The ``Qt`` module and widget
classes (QApplication, QWidget, QVBoxLayout, etc.) are imported
in a unified way to facilitate compatibility across environments.

If neither binding is installed, running this script will raise an
ImportError.  To install PyQt5 use ``pip install pyqt5``.  For PySide2
use ``pip install PySide2``.
"""
try:
    # Prefer PyQt5 if available
    from PyQt5.QtCore import Qt  # type: ignore
    from PyQt5.QtWidgets import (  # type: ignore
        QApplication,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QCheckBox,
        QFileDialog,
        QTextEdit,
        QMessageBox,
    )
except ImportError:
    # Fall back to PySide2
    from PySide2.QtCore import Qt  # type: ignore
    from PySide2.QtWidgets import (  # type: ignore
        QApplication,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QCheckBox,
        QFileDialog,
        QTextEdit,
        QMessageBox,
    )

import image_processing as ip


class ImageProcessingGUI(QWidget):
    """Main window for the image processing GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Intelligent Image Processing")
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()

        # Input directory selection
        input_layout = QHBoxLayout()
        input_label = QLabel("Input directory:")
        self.input_edit = QLineEdit()
        browse_input_btn = QPushButton("Browse…")
        browse_input_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(browse_input_btn)
        layout.addLayout(input_layout)

        # Output directory selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output directory:")
        self.output_edit = QLineEdit()
        browse_output_btn = QPushButton("Browse…")
        browse_output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(browse_output_btn)
        layout.addLayout(output_layout)

        # Options checkboxes
        options_layout = QHBoxLayout()
        self.cb_deskew = QCheckBox("Deskew")
        self.cb_crop = QCheckBox("Crop")
        self.cb_remove_borders = QCheckBox("Remove borders")
        self.cb_denoise = QCheckBox("Denoise")
        options_layout.addWidget(self.cb_deskew)
        options_layout.addWidget(self.cb_crop)
        options_layout.addWidget(self.cb_remove_borders)
        options_layout.addWidget(self.cb_denoise)
        layout.addLayout(options_layout)

        # Run button
        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self.run_processing)
        layout.addWidget(run_btn)

        # Log text area
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit)

        self.setLayout(layout)

    def browse_input(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select input directory")
        if directory:
            self.input_edit.setText(directory)

    def browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select output directory")
        if directory:
            self.output_edit.setText(directory)

    def append_log(self, message: str) -> None:
        self.log_edit.append(message)
        # Scroll to the end
        self.log_edit.moveCursor(self.log_edit.textCursor().End)

    def run_processing(self) -> None:
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Error", "Please select a valid input directory.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please select a valid output directory.")
            return
        os.makedirs(output_dir, exist_ok=True)

        # Gather options
        class Args:
            pass
        args = Args()
        args.deskew = self.cb_deskew.isChecked()
        args.crop = self.cb_crop.isChecked()
        args.remove_borders = self.cb_remove_borders.isChecked()
        args.denoise = self.cb_denoise.isChecked()
        args.verbose = True
        args.border_threshold = 10
        args.crop_margin = 0
        args.denoise_ksize = 3

        # Run in a separate thread to keep UI responsive
        thread = threading.Thread(target=self.process_directory,
                                  args=(input_dir, output_dir, args),
                                  daemon=True)
        thread.start()

    def process_directory(self, input_dir: str, output_dir: str, args) -> None:
        """Run the batch processing and log progress messages."""
        files = [f for f in os.listdir(input_dir)
                 if any(f.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        total = len(files)
        for i, filename in enumerate(files, 1):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            self.append_log(f"[{i}/{total}] Processing {filename}…")
            try:
                image = ip.load_image(input_path)
                processed = ip.process_image(image, args)
                ip.save_image(output_path, processed)
                self.append_log(f"Saved to {output_path}\n")
            except Exception as e:
                self.append_log(f"Error processing {filename}: {e}\n")
        self.append_log("Processing complete.")


def main() -> None:
    app = QApplication([])
    window = ImageProcessingGUI()
    window.resize(600, 400)
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()