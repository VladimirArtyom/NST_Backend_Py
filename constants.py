from enum import Enum
from typing import List
import numpy as np
IMAGENET_MEAN_255: np.ndarray =  np.array([0.485, 0.456, 0.406])
IMAGENET_STD_255: np.ndarray =   np.array([0.229, 0.224, 0.225])

class DefaultConstant(Enum):

    DEFAULT_CONTENT_WEIGHT: float = 1e5
    DEFAULT_STYLE_WEIGHT: float  = 1e4
    DEFAULT_TOTAL_VARIATION_WEIGHT: float = 1e0

    VGG_19: str = "vgg19"
    VGG_16: str = "vgg16"

    O_LBFGS: str = "lbfgs"
    O_ADAM: str = "adam"

    CANVAS_INIT_METHOD_CONTENT: str = "content"
    CANVAS_INIT_METHOD_STYLE: str = "style"
    CANVAS_INIT_METHOD_RANDOM: str = "random"

    SAVING_FREQUENCIES: int = 5

    DEFAULT_CONTENT_IMAGES_DIR: str = "./content_images"
    DEFAULT_STYLE_IMAGES_DIR: str = "./style_images"
    DEFAULT_GENERATED_OUTPUT_DIR: str = "./output_images"\
    
    C_HEIGHT: str = "height"
    C_OPTIMIZER: str = "optimizer"

    
    











