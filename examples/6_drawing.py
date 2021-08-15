from PyQt5 import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class Screen(QMainWindow):
    def setupUi(self):
        self.resize(800, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        self.screen = QLabel(self.centralwidget)
        self.screen.setPixmap(QPixmap("screen.png"))
        self.screen.setScaledContents(True)
        self.screen.setObjectName("screen")

        self.gridLayout.addWidget(self.screen, 0, 0, 1, 1)
        self.setCentralWidget(self.centralwidget)

    def __init__(self):
        super().__init__()
        self.setupUi()
        self.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ui = Screen()
    sys.exit(app.exec_())