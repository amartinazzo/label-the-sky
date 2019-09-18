from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)
import numpy as np
from visualize import plot_bands


def augment(p=.5):
    return Compose([
        # RandomRotate90(),
        Flip(),
        HorizontalFlip(),
        ShiftScaleRotate(shift_limit=0.07, scale_limit=0.3, rotate_limit=45, p=.7),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=90, p=.9),
    ], p=p)


image = np.load('crops/STRIPE82-0001.23020.npy')
print('shape', image.shape)
plot_bands(image, 'STRIPE82-0001.23020.png')

aug = augment(p=0.9)

augmented = np.zeros(image.shape)
augmented = aug(image=image)['image']

plot_bands(augmented, 'STRIPE82-0001.23020-albument.png')
