import math
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def psnr(gt, pred):
    valid = gt - pred
    rmse = math.sqrt(np.mean(valid ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)


class DatasetAB(Dataset):
    H = 256
    W = 455

    def __init__(self, coursedic, finedic, mode='train', enable_augmentation=True):
        super().__init__()
        self.MODE = mode
        self.COURSEDIC = os.path.join(coursedic, self.MODE, 'A') + '/'
        self.FINEDIC = os.path.join(finedic, self.MODE, 'B') + '/'
        self.PNGNAME = self._load_png(self.FINEDIC)
        self.N = len(self.PNGNAME)
        self.enable_augmentation = enable_augmentation and (mode == 'train')

    def __len__(self):
        return self.N

    def _b_to_a_filename(self, b_filename):
        # B: *_500.png / *_500_*  ->  A: *_5.png / *_5_*
        s = b_filename.replace("_500_", "_5_")
        if s == b_filename and "_500.png" in b_filename:
            s = b_filename.replace("_500.png", "_5.png")
        return s

    def _augment(self, image):
        if not self.enable_augmentation:
            return image
        if random.random() > 0.5:
            image = np.fliplr(image)
        if random.random() > 0.5:
            image = np.flipud(image)
        if random.random() > 0.75:
            image = np.rot90(image, k=2)
        return image

    def __getitem__(self, idx):
        fine_filename = self.PNGNAME[idx]
        fine_img = np.array(Image.open(self.FINEDIC + fine_filename), dtype=np.float32)
        course_filename = self._b_to_a_filename(fine_filename)
        course_img = np.array(Image.open(self.COURSEDIC + course_filename), dtype=np.float32)
        if fine_img.shape[0] != self.H or fine_img.shape[1] != self.W:
            fine_img = np.array(Image.fromarray(fine_img).resize((self.W, self.H), Image.LANCZOS))
        if course_img.shape[0] != self.H or course_img.shape[1] != self.W:
            course_img = np.array(Image.fromarray(course_img).resize((self.W, self.H), Image.LANCZOS))
        course_img = (course_img - course_img.min()) / (course_img.max() - course_img.min() + 1e-8)
        fine_img = (fine_img - fine_img.min()) / (fine_img.max() - fine_img.min() + 1e-8)
        course_img = self._augment(course_img)
        fine_img = self._augment(fine_img)
        course = torch.tensor(np.ascontiguousarray(course_img), dtype=torch.float32).unsqueeze(0)
        fine = torch.tensor(np.ascontiguousarray(fine_img), dtype=torch.float32).unsqueeze(0)
        return course, fine, fine_filename

    def _load_png(self, dic):
        return [f.name for f in os.scandir(dic) if f.is_file() and f.name.endswith('.png')]


class DatasetRange(Dataset):
    H = 240
    W = 1376

    def __init__(self, coursedic, finedic, cdepthdic, fdepthdic, comdic, mode='train'):
        super().__init__()
        self.MODE = mode
        self.COURSEDIC = coursedic + '/' + self.MODE + '/'
        self.FINEDIC = finedic + '/' + self.MODE + '/'
        self.CDEPTHDIC = cdepthdic + '/' + self.MODE + '/'
        self.FDEPTHDIC = fdepthdic + '/' + self.MODE + '/'
        self.COMDIC = comdic + '/'
        self.PNGNAME = self._load_png(self.COURSEDIC)
        self.N = len(self.PNGNAME)

    def __len__(self):
        return self.N

    def _intensity_to_depth_filename(self, name):
        return name.replace("intensity", "depth")

    def __getitem__(self, idx):
        course_reflectance = np.array(Image.open(self.COURSEDIC + self.PNGNAME[idx]), dtype=np.float32)
        fine_reflectance = np.array(Image.open(self.FINEDIC + self.PNGNAME[idx]), dtype=np.float32)
        depth_filename = self._intensity_to_depth_filename(self.PNGNAME[idx])
        try:
            course_depth = np.array(Image.open(self.CDEPTHDIC + depth_filename), dtype=np.float32)
            fine_depth = np.array(Image.open(self.FDEPTHDIC + depth_filename), dtype=np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(f"Depth file not found: {depth_filename}")
        course_reflectance = (course_reflectance - np.min(course_reflectance)) / (np.max(course_reflectance) - np.min(course_reflectance) + 1e-8)
        fine_reflectance = (fine_reflectance - np.min(fine_reflectance)) / (np.max(fine_reflectance) - np.min(fine_reflectance) + 1e-8)
        com_reflectance = fine_reflectance.copy()
        course_depth = (course_depth - np.min(course_depth)) / (np.max(course_depth) - np.min(course_depth) + 1e-8)
        fine_depth = (fine_depth - np.min(fine_depth)) / (np.max(fine_depth) - np.min(fine_depth) + 1e-8)
        course = torch.tensor(np.stack([course_reflectance[:, :self.W], course_depth[:, :self.W]], axis=0), dtype=torch.float32)
        fine = torch.tensor(np.stack([fine_reflectance[:, :self.W], fine_depth[:, :self.W]], axis=0), dtype=torch.float32)
        com = torch.tensor(com_reflectance[:, :self.W], dtype=torch.float32)
        return course, fine, com, self.PNGNAME[idx]

    def _load_png(self, dic):
        return [f.name for f in os.scandir(dic) if f.is_file() and f.name.endswith('.png')]


def build_dataset(cfg, split='train'):
    view = cfg.view_type
    data = cfg.data
    if view == 'virtual_camera':
        if not (data.data_root or (data.coursedic and data.finedic)):
            raise ValueError("virtual_camera: set data.data_root or coursedic+finedic")
        coursedic = data.coursedic or os.path.join(data.data_root, getattr(data, 'mode', 'depth'))
        finedic = data.finedic or coursedic
        return DatasetAB(coursedic, finedic, mode=split, enable_augmentation=(split == 'train'))
    if not all([data.coursedic, data.finedic, data.cdepthdic, data.fdepthdic]):
        raise ValueError("panoramic: set coursedic, finedic, cdepthdic, fdepthdic (and optionally comdic)")
    comdic = data.comdic or data.finedic
    return DatasetRange(data.coursedic, data.finedic, data.cdepthdic, data.fdepthdic, comdic, mode=split)
