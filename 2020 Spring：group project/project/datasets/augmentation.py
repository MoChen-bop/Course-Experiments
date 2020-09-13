import numpy as np 
import math
import cv2
import numpy.random as random


class Compose(object):

    def __init__(self, transforms):
        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomMirror(object):

    def __init__(self):
        pass
    
    def __call__(self, image):
        if np.random.randint(2):
            image = np.ascontiguousarray(image[:, ::-1])
            _, width, _ = image.shape
        return image
    

class AugmentColor(object):
    
    def __init__(self):
        self.U = np.array([[-0.56543481, 0.71983482, 0.40240142],
							[-0.5989477, -0.02304967, -0.80036049],
                            [-0.56694071, -0.6935729, 0.44423429]], dtype=np.float32)
        self.EV = np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32)
        self.sigma = 0.1
        self.color_vec = None

    def __call__(self, img):
        color_vec = self.color_vec
        if self.color_vec is None:
            if not self.sigma > 0.0:
                color_vec = np.zeros(3, dtype=np.float32)
            else:
                color_vec = np.random.normal(0.0, self.sigma, 3)
        
        alpha = color_vec.astype(np.float32) * self.EV
        noise = np.dot(self.U, alpha.T) * 255
        return np.clip(img + noise[np.newaxis, np.newaxis, :], 0, 255)


class RandomContrast(object):

    def __init__(self, lower=0.9, upper=1.1):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    
    
    def __call__(self, image):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image = image.astype(np.float32)
            image *= alpha
        return np.clip(image, 0, 255)


class RandomBrightness(object):

    def __init__(self, delta=8):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
    

    def __call__(self, image):
        image = image.astype(np.float32)
        if random.randint(2):
            delta = float(random.uniform(-self.delta, self.delta))
            image += delta
        return np.clip(image, 0, 255)


class Rotate(object):

    def __init__(self, up=15):
        self.up = up
    

    def rotate(self, center, pt, theta):
        xr, yr = center
        yr = -yr
        x, y = pt[:, 0], pt[:, 1]
        y = -y

        theta = theta / 360 * 2 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)
    
        _x = xr + (x - xr) * cos - (y - yr) * sin
        _y = yr + (x - xr) * sin + (y - yr) * cos

        return _x, -_y
    

    def __call__(self, img):
        if np.random.randint(2):
            return img
        angle = np.random.uniform(-self.up, self.up)
        rows, cols = img.shape[0:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])

        return img


class SquarePadding(object):

    def __call__(self, image):
        if np.random.randint(2):
            return image

        if len(image.shape) < 3:
            image = np.stack([image, image, image]).transpose((1, 2, 0))
        H, W, _ = image.shape

        if H == W:
            return image
        
        padding_size = max(H, W)
        expand_image = np.zeros((padding_size, padding_size, 3), dtype=image.dtype)

        if H > W:
            y0, x0 = 0, (H - W) // 2
        else:
            y0, x0 = (W - H) // 2, 0
        
        expand_image[y0:y0+H, x0:x0+W] = image
        image = expand_image

        return image
    

class Padding(object):

    def __init__(self, fill=0):
        self.fill = fill
    

    def __call__(self, image):
        if np.random.randint(2):
            return image
        
        if len(image.shape) < 3:
            image = np.stack([image, image, image]).transpose((1, 2, 0))
        height, width, depth = image.shape

        ratio = np.random.uniform(1, 1.5)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth), dtype=image.dtype)
        expand_image[:, :, :] = self.fill
        expand_image[int(top): int(top + height),
            int(left):int(left + width)] = image
        image = expand_image
        
        return image


class RandomResizeCrop(object):

    def __init__(self, size, scale=(0.9, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio    


    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w
            
            if h < img.shape[0] and w < img.shape[1]:
                j = np.random.randint(0, img.shape[1] - w)
                i = np.random.randint(0, img.shape[0] - h)
                return i, j, h, w
        
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w
    
    
    def __call__(self, image):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        cropped = image[i:i + h, j:j + w, :]
        img = cv2.resize(cropped, self.size)
        return img
    

class RandomResizedLimitCrop(object):
    def __init__(self, size, scale=(0.9, 1.3), ratio=(3. / 4., 4. / 3.)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio


    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)
            
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if np.random.random() < 0.5:
                w, h = h, w
            
            if h < img.shape[0] and w < img.shape[1]:
                j = np.random.randint(0, img.shape[1] - w)
                i = np.random.randint(0, img.shape[0] - h)
                return i, j, h, w
        
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w
        

    def __call__(self, image):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        cropped = image[i:i + h, j:j + w, :]
        scales = np.array([self.size[0] / w, self.size[1] / h])
        
        img = cv2.resize(cropped, self.size)
        return img


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)
    

    def __call__(self, image):
        image = np.array(image).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image


class Resize(object):
    def __init__(self, size=512):
        self.size = size
    
    
    def __call__(self, image):
        if len(image.shape) < 3:
            image = np.stack([image, image, image]).transpose((1, 2, 0))
        h, w, _ = image.shape
        image = cv2.resize(image, (self.size, self.size))
        scales = np.array([self.size / w, self.size / h])
            
        return image
    

class NormChannel(object):
    def __init__(self):
        pass

    def __call__(self, image):
        if len(image.shape) < 3:
            image = np.stack([image, image, image]).transpose((1, 2, 0))

        return image


class Augmentation(object):

    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            NormChannel(),
            # SquarePadding(),
            # Padding(),
            RandomResizeCrop(size),
            RandomResizedLimitCrop(size),
            # Rotate(),
            Resize(size),
            RandomMirror(),
            # RandomBrightness(),
            # RandomContrast(),
            Normalize(mean, std)
        ])
    

    def __call__(self, image):
        return self.augmentation(image)
    

class BaseTransform(object):

    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Resize(size),
            Normalize(mean, std)
        ])

    def __call__(self, image):
        return self.augmentation(image)

