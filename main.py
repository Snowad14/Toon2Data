import numpy as np
import argparse, torch, cv2, glob, warnings; #warnings.filterwarnings("ignore")
from detector import Detector
from inpainter import LamaInpainter
from utils import Utils

parser = argparse.ArgumentParser(description='Extraction of webtoon content')
parser.add_argument('--dirpath', type=str, help='Path of the folder containing the images', required=True)
useCuda = torch.cuda.is_available()
args = parser.parse_args()

print("Loading Models...")
imageDetector = Detector(useCuda)
imageInpainter = LamaInpainter(useCuda)

Utils.createDir("results")

allImages = glob.glob(args.dirpath + "\\*.jpg") + glob.glob(args.dirpath + "\\*.png")

for c, img_path in enumerate(allImages, start=1):
    print(f"\rProcessing Image {c}/{len(allImages)}", end="")

    image = cv2.imread(img_path)
    bubble_mask = imageDetector.detectBubble(image)
    image[(bubble_mask==255)] = [0,0,10]
    text_mask = imageDetector.detectText(image)
    image = imageInpainter._inpaint(image, text_mask)

    for panelId, imgCord in enumerate(imageDetector.detectPanels(image)):
        newPil = image[imgCord[0]:imgCord[0]+(imgCord[2]-imgCord[0]), imgCord[1]:imgCord[1]+(imgCord[3] - imgCord[1])]
        height, width, channels = newPil.shape
        if height < 100 or width < 100: continue # filter too little image
        newMask = bubble_mask[imgCord[0]:imgCord[0]+(imgCord[2]-imgCord[0]), imgCord[1]:imgCord[1]+(imgCord[3] - imgCord[1])]
        if np.sum(newMask) != 0: # is mask filled with black?
            new_mask = cv2.dilate(newMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))) # enlarge the mask because the bubbles are not very well detected
            newPil = imageInpainter._inpaint(newPil, new_mask)
        cv2.imwrite(f"results/{c}-{panelId}.jpg", newPil)

