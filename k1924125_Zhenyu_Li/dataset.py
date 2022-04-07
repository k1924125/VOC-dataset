import cv2, os, math, random, time
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    def __init__(self, Path, target_size=(512, 512), flip=False, random_crop=False, Gaussian_noise=False, adjust_contrast=False, adjust_bright=False, random_rotate= False):
        with open(Path) as f:
            imgPath = list(map(lambda x: x.strip(), f.readlines()))

        self.imgPath_list = np.array(imgPath)
        self.target_size = target_size
        self.indexes = np.arange(len(self.imgPath_list))

        self.flip_ = flip
        self.random_crop_ = random_crop
        self.Gaussian_noise_ = Gaussian_noise
        self.adjust_contrast_ = adjust_contrast
        self.adjust_bright_ = adjust_bright
        self.random_rotate_ = random_rotate

    def __len__(self):
        return len(self.imgPath_list)

    def __getitem__(self, index):
        indexes = self.indexes[index:(index + 1)]

        x, y = self.__data_generation(self.imgPath_list[indexes][0])
        x = np.transpose(x, axes=[2, 0, 1]) / 255.0

        return x, y

    def __data_generation(self, img_path):
        img = cv2.imdecode(np.fromfile('VOCdevkit/VOC2012/JPEGImages/{}.jpg'.format(img_path), np.uint8), cv2.IMREAD_COLOR)
        mask = np.array(Image.open('VOCdevkit/VOC2012/SegmentationClass/{}.png'.format(img_path)))
        mask[mask == 255] = 0

        if np.random.rand(1)[0] > 0.5 and self.flip_:
            img, mask = self.flip(img, mask)
        if np.random.rand(1)[0] > 0.5 and self.random_crop_:
            img, mask = self.random_crop(img, mask)
        if np.random.rand(1)[0] > 0.5 and self.Gaussian_noise_:
            img = self.Gaussian_noise(img)
        if np.random.rand(1)[0] > 0.5 and self.adjust_contrast_:
            img = self.adjust_contrast(img)
        if np.random.rand(1)[0] > 0.5 and self.adjust_bright_:
            img = self.adjust_bright(img)
        if np.random.rand(1)[0] > 0.5 and self.random_rotate_:
            img = self.random_rotate(img)

        img = cv2.resize(img, self.target_size)
        mask = cv2.resize(mask, self.target_size, cv2.INTER_NEAREST)
        return img, mask

    def random_crop(self, img, mask, scale=[0.8, 1.0], ratio=[3. / 4., 4. / 3.]):
        aspect_ratio = math.sqrt(np.random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio
        src_h, src_w = img.shape[:2]

        bound = min((float(src_w) / src_h) / (w ** 2),
                    (float(src_h) / src_w) / (h ** 2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = src_h * src_w * np.random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = np.random.randint(0, src_w - w + 1)
        j = np.random.randint(0, src_h - h + 1)

        img = img[j:j + h, i:i + w]
        mask = mask[j:j + h, i:i + w]
        return img, mask

    def flip(self, img, mask):
        """
        :param mode: 1=Horizontal flip / 0=Vertical flip / -1=Horizontal Vertical flip
        """
        mode = np.random.choice([-1, 0, 1])
        return cv2.flip(img, flipCode=mode), cv2.flip(mask, flipCode=mode)

    def random_rotate(self, img, angle_range=(-45, 45)):
        """
        :param angle_range:  Range of rotation angles (min,max)   >0 indicates counterclockwise，
        :return:
        """
        height, width = img.shape[:2]  
        center = (width / 2, height / 2)
        angle = random.randrange(*angle_range, 1)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (height, width))
        return img

    def Gaussian_noise(self, img, rand_range=(5, 40)):
        img = np.asarray(img, np.float)
        sigma = random.randint(*rand_range)
        nosie = np.random.normal(0, sigma, size=img.shape)
        img += nosie
        img = np.uint8(np.clip(img, 0, 255))
        return img

    def adjust_contrast(self, img):
        contrast = np.random.rand(1)[0] * 1.2
        return np.uint8(np.clip((contrast * img), 0, 255))

    def adjust_bright(self, img):
        brightness = np.random.rand(1)[0] * 100
        return np.uint8(np.clip((img + brightness), 0, 255))

if __name__ == '__main__':
    import tqdm

    a = DataGenerator('VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt')
    idx = 0
    for i, j in tqdm.tqdm(a):
        print(i.shape, j.shape)
    print(idx)
