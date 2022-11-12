import numpy as np
import craft_text_detector, cv2, shutil, os

class Utils:

    @staticmethod
    def createDir(dirpath : str):
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
        os.mkdir(dirpath)

    @staticmethod
    def changeColor(img : np.ndarray, colorInput : tuple, colorOutput : tuple) -> np.ndarray:
        r1, g1, b1 = colorInput[0], 0, 10
        r2, g2, b2 = 0, 0, 0

        red, green, blue = img[:,:,0], img[:,:,1], img[:,:,2]
        mask = (red == r1) & (green == g1) & (blue == b1)
        img[:,:,:3][mask] = [r2, g2, b2]
        return img
    
    @staticmethod
    def do_bboxes_overlap(a, b):
        return (
            a[0] < b[2] and
            a[2] > b[0] and
            a[1] < b[3] and
            a[3] > b[1]
        )

    @staticmethod
    def merge_bboxes(a, b):
        return (
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3])
        )

    @staticmethod
    def drawSegmentation(img, result):
        image = craft_text_detector.read_image(img)

        for i, region in enumerate(result):
            region = np.array(region).astype(np.int32).reshape((-1))

            region = region.reshape(-1, 2)
            cv2.fillPoly(
                image,
                [region.reshape((-1, 1, 2))],
                color=(255,255,255)
            )
        return image