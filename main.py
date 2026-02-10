import argparse
import constants as c
import torch

import numpy as np
from typing import Dict
from torch import Tensor
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from torch.optim.lbfgs import LBFGS
import losses as l
import utils as u
import matplotlib.pyplot as plt

import os
def process_image(model, optimizer: Optimizer, original_representations,
                 content_feature_maps_index, style_feature_maps_indices):
    
    def step(generated_image):
        total_loss, content_loss, style_loss, tv_loss = l.calculate_loss(model, generated_img=generated_image,
                                                                         original_representations=original_representations, content_feature_index=content_feature_maps_index,
                                                                         style_feature_indices=style_feature_maps_indices)
        
        total_loss.backward() # calculates the gradients of total_loss with respect to image
        optimizer.step() 
        optimizer.zero_grad()

        return total_loss, content_loss, style_loss, tv_loss

    return step

def nst(content_filename: str, style_filename: str, init_method: str, height: int, optimizer: str = c.DefaultConstant.O_LBFGS.value):
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_image_path: str = os.path.join(c.DefaultConstant.DEFAULT_CONTENT_IMAGES_DIR.value, content_filename) 
    style_image_path: str = os.path.join(c.DefaultConstant.DEFAULT_STYLE_IMAGES_DIR.value, style_filename)

    output_images_path: str = c.DefaultConstant.DEFAULT_GENERATED_OUTPUT_DIR.value
    
    content_image: np.ndaraay = u.preprocess_image(content_image_path, height, device)
    style_image: np.ndarray = u.preprocess_image(style_image_path, height, device)

    if init_method == c.DefaultConstant.CANVAS_INIT_METHOD_RANDOM.value:
        gaussian_noise_image: np.ndarray = np.random.normal(loc=0, scale=90, size=content_image.shape).astype(np.float32)
        initialize_image = torch.from_numpy(gaussian_noise_image).float().to(device)

    elif init_method == c.DefaultConstant.CANVAS_INIT_METHOD_CONTENT.value:
        initialize_image  = content_image
    else:
        style_image_resize = u.preprocess_image(style_image_path, content_image.shape[:2], device) #Takes height and channel
        initialize_image = style_image_resize

    generated_image = Variable(initialize_image, requires_grad=True)
    
    nn_model, content_indx_et_l_name, style_indices_et_l_name = u.prepare_model(c.DefaultConstant.VGG_16.value, device)

    set_of_content_feature_maps = nn_model(content_image)
    set_of_style_feature_maps = nn_model(style_image)

    original_content_representation_feature_map = set_of_content_feature_maps[content_indx_et_l_name[0]].squeeze(axis=0) # Removing the batch from the layer
    original_style_representation_feature_maps = [l.gram_matrix(feature) for cnt, feature in enumerate(set_of_style_feature_maps) if cnt in style_indices_et_l_name[0]]
    
    original_representations = [original_content_representation_feature_map, original_style_representation_feature_maps]
    
    num_of_iterations: Dict = {
        c.DefaultConstant.O_LBFGS.value: 1000,
        c.DefaultConstant.O_ADAM.value: 3000
    }

    if optimizer == c.DefaultConstant.O_ADAM.value:
        optimizer_obj = Adam((generated_image,), lr=1e1)
        step_func = process_image(nn_model, optimizer_obj,
                                original_representations, content_indx_et_l_name[0],
                                style_indices_et_l_name[0])
        for iteration in range(num_of_iterations[c.DefaultConstant.O_ADAM.value]):
            total_loss, content_loss, style_loss, tv_loss = step_func(generated_image)
            with torch.no_grad():
                print(f"Adam | ieration: {iteration:03}, total_loss: {total_loss.item():12.4f}, content_loss: {content_loss.item():12.4f}, style_loss: {style_loss.item():12.4f}, tv_loss: {tv_loss.item():12.4f}")
                if iteration % 50 == 0 :
                    file_name: str =os.path.join(output_images_path, str(iteration)+".jpg")
                    u.save_image_(generated_image, file_name)
    elif optimizer == c.DefaultConstant.O_LBFGS.value:
        optimizer = LBFGS((generated_image,), max_iter=num_of_iterations[c.DefaultConstant.O_LBFGS.value], line_search_fn='strong_wolfe')
        iteration = 0

        def closure():
            nonlocal iteration
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = l.calculate_loss(nn_model, generated_img=generated_image,
                                                                         original_representations=original_representations, content_feature_index=content_indx_et_l_name[0],
                                                                         style_feature_indices=style_indices_et_l_name[0])
            total_loss.backward()
            with torch.no_grad():
                print(f"Adam | ieration: {iteration:03}, total_loss: {total_loss.item():12.4f}, content_loss: {content_loss.item():12.4f}, style_loss: {style_loss.item():12.4f}, tv_loss: {tv_loss.item():12.4f}")
                if iteration % 50 == 0 :
                    file_name: str =os.path.join(output_images_path, str(iteration)+".jpg")
                    u.save_image_(generated_image, file_name)
            iteration += 1
            return total_loss

        optimizer.step(closure)
    
    
if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument("--content_image_path", type=str,
                         help="Content image path", default="")
    parser.add_argument("--style_image_path", type=str,
                        help="Style image path", default="")
    parser.add_argument("--content_weight", type=float, default=c.DefaultConstant.DEFAULT_CONTENT_WEIGHT.value)
    parser.add_argument("--style_weight", type=float, default=c.DefaultConstant.DEFAULT_STYLE_WEIGHT.value)
    parser.add_argument("--tv_weight", type=float, default=c.DefaultConstant.DEFAULT_TOTAL_VARIATION_WEIGHT.value)

    parser.add_argument("--height", type=float, default=400)

    parser.add_argument("--optimizer", type=str, choices=[c.DefaultConstant.O_LBFGS.value, c.DefaultConstant.O_ADAM.value],
                         default=c.DefaultConstant.O_LBFGS.value)    
    parser.add_argument("--model", type=str, choices=[c.DefaultConstant.VGG_19.value,
                                                      c.DefaultConstant.VGG_16.value])
    parser.add_argument("--init_method", type=str, choices=[c.DefaultConstant.CANVAS_INIT_METHOD_STYLE.value,
                                                           c.DefaultConstant.CANVAS_INIT_METHOD_RANDOM.value,
                                                           c.DefaultConstant.CANVAS_INIT_METHOD_CONTENT.value],
                                                           default=c.DefaultConstant.CANVAS_INIT_METHOD_RANDOM.value)
    parser.add_argument("--device", type=str, default="cpu")
    args: Dict = parser.parse_args()
    #nst("kk.jpg","memes.jpg",c.DefaultConstant.CANVAS_INIT_METHOD_CONTENT.value, 400, optimizer=c.DefaultConstant.O_ADAM.value) 
    #print(c.DefaultConstant.IMAGENET_MEAN_255.value)

    img = (u.preprocess_image("./content_images/kk.jpg",400 , device=args.device))
    u.save_image_(img.squeeze(0),"why.jpg")