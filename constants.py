from enum import Enum
from typing import List
import numpy as np
IMAGENET_MEAN_255: np.ndarray =  np.array([0.485, 0.456, 0.406])
IMAGENET_STD_255: np.ndarray =   np.array([0.229, 0.224, 0.225])

class DefaultConstant(Enum):

    DEFAULT_CONTENT_WEIGHT: float = 1e5
    DEFAULT_STYLE_WEIGHT: float  = 1e4
    DEFAULT_TOTAL_VARIATION_WEIGHT: float = 1e0

    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_NUM_EPOCHS: int = 20
    DEFAULT_TRAINING_LR: float = 1e-4

    VGG_19: str = "vgg19"
    VGG_16: str = "vgg16"
    VGG_16_EXPERIMENTAL: str ="vgg_16_experimental"
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


    DEFAULT_SIZE_IMAGE_1: int = 256
    DEFAULT_SIZE_IMAGE_2: int = 512

    DROPOUT_DEFAULT: float = 0.5
    LINEAR_DEFAULT: int = 4096
    
    SCHEDULER_FACTOR_DEFAULT: float = 0.5
    SCHEDULER_PATIENCE_DEFAULT: float = 3

    M_TRAIN_LOSS: str = "train_loss"
    M_TRAIN_ACCURACY: str = "train_accuracy"
    M_TRAIN_RECALL: str = "train_recall" 
    M_TRAIN_F1: str = "train_f1"
    M_TRAIN_MCC: str = "train_mcc"
    M_TRAIN_PREC: str = "train_precision"
    M_TRAIN_SUPPORT: str ="train_support"

    M_TEST_LOSS: str = "test_loss"
    M_TEST_ACCURACY: str = "test_acurracy" 
    M_TEST_RECALL: str = "test_recall"
    M_TEST_F1: str = "test_f1"
    M_TEST_MCC: str = "test_mcc"
    M_TEST_PREC: str = "test_precision"
    M_TEST_SUPPORT: str = "test_support"








