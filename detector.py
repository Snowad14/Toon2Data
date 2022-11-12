from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import dilation
from skimage.measure import label
from skimage.color import label2rgb
from skimage.measure import regionprops
from scipy import ndimage as ndi
from torch.utils.data import DataLoader
from utils import Utils
import numpy as np
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import craft_text_detector, torch, cv2, copy

class Detector:
    
    def __init__(self, cuda : bool):
        self.cuda = cuda
        self.device = "cuda" if cuda else "cpu"
        self.pattern_path ='./pattern.png'
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.__loadModels()
    
    def __loadModels(self):
        self.bubbleModel = torch.load("models/train_model_mob_trans_4.pth")
        self.bubbleModel = self.bubbleModel.to(torch.device(self.device))
        self.refine_net = craft_text_detector.load_refinenet_model(cuda=self.cuda, weight_path="models/refine_net.pth")
        self.craft_net = craft_text_detector.load_craftnet_model(cuda=self.cuda, weight_path="models/craft_net.pth")
    
    def detectText(self, img : np.ndarray) -> np.ndarray:
        image = craft_text_detector.read_image(img)

        prediction_result = craft_text_detector.get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=self.cuda,
            long_size=1280
        )

        mask = Utils.drawSegmentation(np.zeros(img.shape[:2], dtype="uint8"), prediction_result["boxes"])
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        return mask

    def detectBubble(self, img : np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(img).convert('RGB')
        width, height = pil_image.size

        pattern = cv2.imread(self.pattern_path)
        pattern = cv2.resize(pattern,(width,height))
        null_img = np.ones((width,height, 3), dtype=np.uint8)
        pil_image = pil_image.resize((width, height))
        mask = np.zeros((width, height))

        tensor_image = self.transform(pil_image)
        tensor_image = tensor_image.to(self.device).unsqueeze(0)
        pr_mask = self.bubbleModel.predict(tensor_image)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        pr_mask_pil = Image.fromarray(np.uint8(pr_mask))
        pr_mask_pil = pr_mask_pil.resize((width, height), Image.NEAREST)

        segmentation_mask = np.array(pr_mask_pil)
        segmentation_mask = segmentation_mask.reshape((height, width, 1)) * 255
        segmentation_mask = segmentation_mask.astype(np.uint8)
        mask = copy.deepcopy(segmentation_mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        segmentation_mask = segmentation_mask.squeeze(axis=2)
        mask[segmentation_mask == 255] = pattern[segmentation_mask == 255]
        return segmentation_mask


    def detectPanels(self, img : np.ndarray):
        # https://maxhalford.github.io/blog/comic-book-panel-segmentation/#canny-edge-detection

        img = Utils.changeColor(img, (0, 0, 10), (0, 0, 0))

        grayscale = rgb2gray(img)
        edges = canny(grayscale)
        thick_edges = dilation(dilation(edges))
        segmentation = ndi.binary_fill_holes(thick_edges)
        labels = label(segmentation)

        regions = regionprops(labels)
        panels = []

        for region in regions:
            for i, panel in enumerate(panels):
                if Utils.do_bboxes_overlap(region.bbox, panel):
                    panels[i] = Utils.merge_bboxes(panel, region.bbox)
                    break
            else:
                panels.append(region.bbox)

        for i, bbox in reversed(list(enumerate(panels))):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < 0.01 * img.shape[0] * img.shape[1]:
                del panels[i]

        return panels