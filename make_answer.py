import os
import re
import sys

import cv2
import numpy as np
import pymysql
import pytesseract
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit

eng = re.compile('[a-zA-Z]{6}')

HOST = 'YOUR DB HOST'
pytesseract.pytesseract.tesseract_cmd = 'tesseract PATH'

def next_image():
    file_list = os.listdir('./data')
    file_list.sort(key=lambda f: int(f.split('.')[0]))
    for image in file_list:
        yield image


image_generator = next_image()

def alpha_to_gray(img):
    alpha_channel = img[:, :, 3]
    _, mask = cv2.threshold(alpha_channel, 128, 255, cv2.THRESH_BINARY)  # binarize mask
    color = img[:, :, :3]
    img = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sharpening(img):
    kernel_sharpen_1 = np.array(
        [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # 정규화를 하지 않은 이유는 모든 값을 다 더하면 1이되기때문에 1로 나눈것과 같은 효과

    # applying different kernels to the input image
    output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
    return output_1


def ocr(img):
    t = pytesseract.image_to_string(img, lang='eng',
                                    config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return t


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Make Answer'
        self.left = 200
        self.top = 200
        self.width = 640
        self.height = 480
        self.init_ui()
        self.conn = pymysql.connect(host=HOST, user='uploader', password='', db='data', charset='utf8')
        self.curs = self.conn.cursor()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create widget
        self.label = QLabel(self)
        self.label2 = QLabel(self)

        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.resize(280, 40)
        self.textbox.move((self.size().width() - self.textbox.size().width()) // 2, 400)
        self.textbox.returnPressed.connect(self.search_slot)

        self.next_image()
        self.show()

    def next_image(self):
        self.image = next(image_generator)
        self.pixmap = QPixmap('./data/' + self.image)
        self.pixmap = self.pixmap.scaled(self.pixmap.size().width() * 2, self.pixmap.size().height() * 2)
        self.label.setPixmap(self.pixmap)
        self.label.move((self.size().width() - self.pixmap.size().width()) // 2, 200)

        image = cv2.imread('./data/' + self.image, cv2.IMREAD_UNCHANGED)
        img = alpha_to_gray(image)

        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.medianBlur(img, 3)
        kernel = np.ones((4, 4), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)

        height, width = img.shape
        qImg = QImage(img.data, width, height, width, QImage.Format_Grayscale8)
        self.pixmap2 = QPixmap.fromImage(qImg)
        self.pixmap2 = self.pixmap2.scaled(self.pixmap2.size().width() * 2, self.pixmap2.size().height() * 2)
        self.label2.setPixmap(self.pixmap2)
        self.label2.move((self.size().width() - self.pixmap2.size().width()) // 2, 50)

        self.textbox.setText(ocr(img))

    @pyqtSlot()
    def search_slot(self):
        text = self.textbox.text()
        text = text.upper()
        if eng.match(text):
            sql = """INSERT INTO data.answer(data_no, answer)
                     VALUES (%s, %s)"""
            self.curs.execute(sql, (self.image.split('.')[0], text))
            self.conn.commit()
            os.remove('./data/' + self.image)
            self.next_image()

    def closeEvent(self, event):
        # do stuff
        self.conn.close()
        event.accept()  # let the window close


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
