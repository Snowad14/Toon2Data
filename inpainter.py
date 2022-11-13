# https://github.com/dmMaze/BallonsTranslator/blob/v1.3.14/ballontranslator/dl/inpaint/__init__.py

import numpy as np
import cv2
import torch
from typing import Dict, List
from models.lama import load_lama_mpe, resize_keepasp, LamaFourier

LAMA_MPE: LamaFourier = None

class LamaInpainter():

    inpaint_size = 1024

    def __init__(self, withcuda):
        self.device = "cuda" if withcuda else "cpu"
        global LAMA_MPE
        if LAMA_MPE is None:
            self.model = LAMA_MPE = load_lama_mpe(r'models/lama_mpe.ckpt', self.device)
        else:
            self.model = LAMA_MPE
            self.model.to(self.device)
        self.inpaint_by_block = True if self.device == 'cuda' else False
        self.inpaint_size = 1024

    def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        new_shape = self.inpaint_size if max(img.shape[0: 2]) > self.inpaint_size else None
        # high resolution input could produce cloudy artifacts
        img = resize_keepasp(img, new_shape, stride=64)
        mask = resize_keepasp(mask, new_shape, stride=64)

        im_h, im_w = img.shape[:2]
        longer = max(im_h, im_w)
        pad_bottom = longer - im_h if im_h < longer else 0
        pad_right = longer - im_w if im_w < longer else 0
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 255.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1
        rel_pos, _, direct = self.model.load_masked_position_encoding(mask_torch[0][0].numpy())
        rel_pos = torch.LongTensor(rel_pos).unsqueeze_(0)
        direct = torch.LongTensor(direct).unsqueeze_(0)

        if self.device != 'cpu':
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
            rel_pos = rel_pos.to(self.device)
            direct = direct.to(self.device)
        img_torch *= (1 - mask_torch)
        return img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right

    @torch.no_grad()
    def _inpaint(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
        img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
        
        img_inpainted = (img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if pad_bottom > 0:
            img_inpainted = img_inpainted[:-pad_bottom]
        if pad_right > 0:
            img_inpainted = img_inpainted[:, :-pad_right]
        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != im_h or new_shape[1] != im_w :
            img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
        return img_inpainted



