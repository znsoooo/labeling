import os
import tkinter

import cv2
import numpy as np

__title__ = 'Labeling'

root = tkinter.Tk()
root.withdraw()
SCREEN_WIDTH = root.winfo_screenwidth() * 0.8
SCREEN_HEIGHT = root.winfo_screenheight() * 0.8
root.destroy()


def read(path):
    with open(path, 'a+', encoding='u8') as f:
        f.seek(0)
        return f.read()


def write(path, data):
    if isinstance(data, list):
        data = '\n'.join(data)
    with open(path, 'w', encoding='u8') as f:
        f.write(data)
    return data


def ReadLabel(path, img_wh):
    w0, h0 = img_wh
    boxes = []
    for line in read(path).splitlines():
        id, x, y, w, h = map(float, line.split())
        x1, y1, x2, y2 = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        boxes.append([int(id), x1 * w0, y1 * h0, x2 * w0, y2 * h0])
    return boxes


def SaveLabel(path, img_wh, boxes):
    w0, h0 = img_wh
    lines = []
    for id, x1, y1, x2, y2 in boxes:
        x, y, w, h = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        line = f'{id} {x / w0} {y / h0} {w / w0} {h / h0}'
        lines.append(line)
    return write(path, lines)


def ReadImage(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), -1)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w, n = img.shape
    k = min(SCREEN_WIDTH / w, SCREEN_HEIGHT / h)
    img = cv2.resize(img, (int(w * k), int(h * k)))
    return img


def Hsv2Bgr(h=1.0, s=1.0, v=1.0):
    return cv2.cvtColor(np.uint8([[[h * 180, s * 255, v * 255]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()


def DrawBox(img, rect, label='', color=(0, 0, 255)):
    x1, x2 = sorted(map(int, rect[0::2]))
    y1, y2 = sorted(map(int, rect[1::2]))
    if x1 != x2 and y1 != y2:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        fontFace, fontScale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        fw, fh = cv2.getTextSize(label, fontFace, fontScale, thickness)[0]
        cv2.rectangle(img, (x1 - 1, y1), (x1 + fw + 1, y1 - fh - 4), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), fontFace, fontScale, (255, 255, 255), thickness)
    return img


class Labeling:
    def __init__(self, folder):
        self.img_folder = os.path.join(folder, 'images')
        self.lab_folder = os.path.join(folder, 'labels')

        assert os.path.isdir(self.img_folder), f'Image folder should be exist: "{self.img_folder}"'
        os.makedirs(self.lab_folder, exist_ok=True)

        self.files = os.listdir(self.img_folder)
        self.idx = 0
        self.NextImage()

        cv2.namedWindow(__title__)
        cv2.setMouseCallback(__title__, self.MouseEvent)
        key = 0
        while key != -1:
            key = cv2.waitKeyEx(0)
            self.KeyEvent(key)

    @property
    def img_path(self):
        return os.path.join(self.img_folder, self.files[self.idx])

    @property
    def lab_path(self):
        return os.path.join(self.lab_folder, os.path.splitext(self.files[self.idx])[0] + '.txt')

    @property
    def img_wh(self):
        return self.img.shape[1::-1]

    def MouseEvent(self, key, x, y, flag, param):
        # print(f'MouseEvent: {(evt, x, y, flag)}')
        if key == 0 and flag == 1:
            self.OnLeftDrag(x, y)
        elif key == 1:
            self.OnLeftDown(x, y)
        elif key == 4:
            self.OnLeftUp(x, y)
        elif key == 2:
            self.OnRightDown()
        elif key == 10:
            self.NextImage(1 if flag < 0 else -1)

    def KeyEvent(self, key):
        # print(f'KeyEvent: {key} ({hex(key)})')
        if key == 13:
            os.popen('explorer /select, "%s"' % os.path.abspath(self.img_path))
        elif key == 27:
            cv2.destroyAllWindows()
        elif key in [0x210000, 0x250000, 0x260000]:
            self.NextImage(-1)
        elif key in [0x220000, 0x270000, 0x280000]:
            self.NextImage(1)

    def DrawBoxes(self):
        img = self.img.copy()
        for i, (id, *rect) in enumerate(self.boxes):
            DrawBox(img, rect, str(id), Hsv2Bgr(id / 10))
        cv2.imshow(__title__, img)

    def NextImage(self, next=0):
        self.idx = (self.idx + next) % len(self.files)
        self.img = ReadImage(self.img_path)
        self.ReadLabel()

    def OnLeftDown(self, x, y):
        self.boxes.append([0, x, y, x, y])
        self.DrawBoxes()

    def OnLeftDrag(self, x, y):
        self.boxes[-1][-2:] = [x, y]
        self.DrawBoxes()

    def OnLeftUp(self, x, y):
        self.boxes[-1][-2:] = [x, y]
        self.SaveLabel()

    def OnRightDown(self):
        self.boxes = self.boxes[:-1]
        self.SaveLabel()

    def ReadLabel(self):
        self.boxes = ReadLabel(self.lab_path, self.img_wh)
        self.DrawBoxes()

    def SaveLabel(self):
        SaveLabel(self.lab_path, self.img_wh, self.boxes)
        self.DrawBoxes()


if __name__ == '__main__':
    Labeling('datasets/coco128')
