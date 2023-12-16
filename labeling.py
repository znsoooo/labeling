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


def label2xyxy(text, wh):
    x, y, w, h = map(float, text.split()[1:])
    xyxy = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
    return [k * y for k, y in zip(xyxy, wh * 2)]


def xyxy2label(xyxy, wh):
    x1, y1, x2, y2 = xyxy
    xywh = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return '0 ' + ' '.join(str(x / y) for x, y in zip(xywh, wh * 2))


test_xyxy, test_wh = [2, 3, 5, 6], [11, 13]
test_xyxy2 = label2xyxy(xyxy2label(test_xyxy, test_wh), test_wh)
assert test_xyxy == test_xyxy2, test_xyxy2


class Labeling:
    def __init__(self, folder):
        self.img_folder = os.path.join(folder, 'images')
        self.lab_folder = os.path.join(folder, 'labels')
        os.makedirs(self.lab_folder, exist_ok=True)

        self.names = os.listdir(self.img_folder)
        self.idx = 0
        self.SetImage()

        cv2.namedWindow(__title__)
        cv2.setMouseCallback(__title__, self.MouseLoop)
        self.KeyLoop()

    @property
    def img_path(self):
        return os.path.join(self.img_folder, self.names[self.idx])

    @property
    def lab_path(self):
        return os.path.join(self.lab_folder, os.path.splitext(self.names[self.idx])[0] + '.txt')

    def MouseLoop(self, evt, x, y, flag, param):
        # print((evt, x, y, flag))
        if evt == 0 and flag == 1:
            self.OnLeftDrag(x, y)
        elif evt == 1:
            self.OnLeftDown(x, y)
        elif evt == 4:
            self.OnLeftUp(x, y)
        elif evt == 2:
            self.OnRightDown()
        elif evt == 10:
            self.SetImage(1 if flag < 0 else -1)

    def KeyLoop(self):
        while True:
            key = cv2.waitKey(0)
            if key == 13:
                os.popen('explorer /select, "%s"' % os.path.abspath(self.img_path))
            elif key == 27:
                return cv2.destroyAllWindows()

    def DrawRects(self):
        img = self.img.copy()
        for i, (x1, y1, x2, y2) in enumerate(self.rects):
            color = (0, 255, 0) if i == len(self.rects) - 1 else (255, 0, 0)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.imshow(__title__, img)

    def SetImage(self, next=0):
        self.idx = (self.idx + next) % len(self.names)
        img = cv2.imdecode(np.fromfile(self.img_path, np.uint8), -1) # read from unicode path
        h, w, n = img.shape
        self.k = k = min(SCREEN_WIDTH / w, SCREEN_HEIGHT / h)
        self.img = cv2.resize(img, (int(w * k), int(h * k)))
        self.ReadLabels()

    def OnLeftDown(self, x, y):
        self.rects.append([x, y, x, y])
        self.DrawRects()

    def OnLeftDrag(self, x, y):
        self.rects[-1][2:] = [x, y]
        self.DrawRects()

    def OnLeftUp(self, x, y):
        self.rects[-1][2:] = [x, y]
        self.SaveLabels()

    def OnRightDown(self):
        self.rects = self.rects[:-1]
        self.SaveLabels()

    def ReadLabels(self):
        with open(self.lab_path, 'a+') as f:
            f.seek(0)
            self.rects = [label2xyxy(label, self.img.shape[1::-1]) for label in f.read().splitlines()]
        self.DrawRects()

    def SaveLabels(self):
        with open(self.lab_path, 'w') as f:
            f.write('\n'.join(xyxy2label(rect, self.img.shape[1::-1]) for rect in self.rects))
        self.DrawRects()


if __name__ == '__main__':
    Labeling('datasets/coco128')
