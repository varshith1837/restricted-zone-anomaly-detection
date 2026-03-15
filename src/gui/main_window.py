"""
Sentinel Vision — Main Dashboard Window
==========================================
PyQt5 main window with sidebar navigation.
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QHBoxLayout, QVBoxLayout, QFrame,
                              QLabel, QPushButton, QStackedWidget)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from src.gui.registration_page import RegistrationPage
from src.gui.surveillance_page import SurveillancePage


class MainDashboard(QMainWindow):
    """Main application window with sidebar navigation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentinel Vision — AI Security System")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SIDEBAR ---
        sidebar = QFrame()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("""
            QFrame { background-color: #1e1e2e; border-right: 1px solid #444; }
            QPushButton {
                background-color: transparent; color: #bbb; border: none;
                padding: 15px; text-align: left; font-size: 16px;
            }
            QPushButton:hover { background-color: #333; color: white; }
            QPushButton:checked { background-color: #00adb5; color: white; font-weight: bold; }
        """)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        lbl_title = QLabel("SENTINEL\nVISION")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("padding: 30px; font-size: 22px; font-weight: bold; color: #00adb5;")
        sidebar_layout.addWidget(lbl_title)

        lbl_sub = QLabel("AI-Powered Security")
        lbl_sub.setAlignment(Qt.AlignCenter)
        lbl_sub.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 20px;")
        sidebar_layout.addWidget(lbl_sub)

        self.btn_register = QPushButton("  👤  Register Face")
        self.btn_register.setCheckable(True)
        self.btn_register.clicked.connect(lambda: self.switch_page(0))

        self.btn_surveillance = QPushButton("  📹  Surveillance")
        self.btn_surveillance.setCheckable(True)
        self.btn_surveillance.clicked.connect(lambda: self.switch_page(1))

        self.btn_exit = QPushButton("  ❌  Exit System")
        self.btn_exit.clicked.connect(self.close)

        sidebar_layout.addWidget(self.btn_register)
        sidebar_layout.addWidget(self.btn_surveillance)
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(self.btn_exit)

        # --- CONTENT ---
        self.stacked = QStackedWidget()
        self.page_register = RegistrationPage()
        self.page_surveillance = SurveillancePage()
        self.stacked.addWidget(self.page_register)
        self.stacked.addWidget(self.page_surveillance)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stacked)

        self.btn_register.setChecked(True)
        self.stacked.setCurrentIndex(0)

    def switch_page(self, index):
        if index == 0:
            self.page_surveillance.stop_system()
            self.page_register.start_camera()
            self.btn_register.setChecked(True)
            self.btn_surveillance.setChecked(False)
        elif index == 1:
            self.page_register.stop_camera()
            self.page_surveillance.start_system()
            self.btn_register.setChecked(False)
            self.btn_surveillance.setChecked(True)
        self.stacked.setCurrentIndex(index)

    def closeEvent(self, event):
        self.page_register.stop_camera()
        self.page_surveillance.stop_system()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainDashboard()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
