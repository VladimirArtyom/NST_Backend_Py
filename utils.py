import os
import numpy as np
import cv2 
from torchvision import transforms
from torch import Tensor
from torchvision.utils import save_image

import constants as c

from models.definitions.vgg_nets import Vgg16, Vgg19, Vgg16Experimental


def prepare_model(model, device: str, experimental: bool = False):
    if model == c.DefaultConstant.VGG_16.value:
        if experimental:
            model = Vgg16Experimental(requires_grad=False, show_progress=True)
        else:
            model = Vgg16(requires_grad=False, show_progress=True)

    elif model == c.DefaultConstant.VGG_19.value:
        model = Vgg19(requires_grad=False, show_progress=True)
    else:
        raise ValueError(f'{model} is not supported yet')

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names

    content_indx_et_l_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_indices_et_l_names =   (style_feature_maps_indices, layer_names)

    return model.to(device).eval(), content_indx_et_l_name, style_indices_et_l_names


def load_image(image_path: str, target_shape: int = None) -> np.ndarray:
    if not os.path.exists(image_path):
        raise Exception(f"This path {image_path} doesn't exist")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR to RGB

    if target_shape is not None:
        if isinstance(target_shape, int):
            image_height, image_width = image.shape[:2]
            n_height: int = target_shape
            n_width: int = int(image_width * (n_height / image_height))

            image = cv2.resize(image, (n_width, n_height), interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)

    image = image.astype(np.float32)
    image /= 255.0
    return image


def preprocess_image(img_path: str, target_shape, device: str) :
    image = load_image(img_path, target_shape)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=c.IMAGENET_MEAN_255,
                             std=c.IMAGENET_STD_255,
                              inplace=True),
        transforms.Lambda(lambda pixel: pixel.mul(255.))
    ])
    image = transform(image).to(device).unsqueeze(0) # add batch 
    return image

def save_image_(image: Tensor, ):
    transform = transforms.Compose([
        transforms.Lambda(lambda pixel: pixel.div(255.)),
        transforms.Normalize((-1*c.IMAGENET_MEAN_255/c.IMAGENET_STD_255),
                             (1.0/c.IMAGENET_STD_255), inplace=True)
    ])

    image = transform(image)
    save_image(image)
def show_tensor_image(image: Tensor, filename: str = None):
    transform = transforms.Compose([
        transforms.Lambda(lambda pixel: pixel.div(255.)),
        transforms.Normalize((-1 * c.IMAGENET_MEAN_255/ c.IMAGENET_STD_255),
                             (1.0 / c.IMAGENET_STD_255), inplace=True)
    ])
    image = transform(image.cpu().squeeze(axis=0))

    if filename is not None:
        save_image(image, filename)

    return image.permute(1, 2, 0).detach().numpy()
