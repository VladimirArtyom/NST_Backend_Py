import metrics as mi
from typing import List, Dict
from torch import Tensor, from_numpy
import utils as u
import os

def get_calculate_the_metrics(original_image: Tensor, generated_image: Tensor,
                              device: str= "cpu", pixel_range: float = 1.0, normalize: bool = True,
                              weight_path: str = None):
    
    ssi: Tensor = mi.structural_similarity_index(original_image, generated_image, pixel_range=pixel_range, normalize=normalize)
    if weight_path:
        lpips: Tensor = mi.learned_perceptual_image_patch_similarity(original_image, generated_image,normalize=normalize, weight_path=weight_path)
    else:
        lpips: Tensor = mi.learned_perceptual_image_patch_similarity(original_image, generated_image,normalize=normalize)
    mse: Tensor = mi.mean_squared_error(original_image, generated_image, device)
    psnt: Tensor = mi.peak_signal_to_noise_ratio(original_image, generated_image,pixel_range=pixel_range,normalized=normalize)

    print(f"The SSI metric value is {ssi.item()}")
    print(f"The LPIPS metric value is {lpips.item()}")
    print(f"The MSE metric value is {mse.item()}")
    print(f"The PSNT metric value is {psnt.item()}")


if __name__ == "__main__":
    original_content_image: str = "./content_images/kk.jpg"
    generated_image_vgg_trained: str ="./output_images/3000_steps_with_retraining_VGG19.jpg"
    generated_image_vgg_original: str ="./output_images/3000_steps_without_retraining_VGG19.jpg"
    model_weight_path: str = "./saved_model/batik_vgg19_features.pth"

    img_ori = from_numpy(u.load_image(original_content_image,(1000, 1000)))

    img_gen_trained_vgg19 = from_numpy(u.load_image(generated_image_vgg_trained, (1000, 1000)))
    img_gen_original_vgg19 = from_numpy(u.load_image(generated_image_vgg_original, (1000, 1000)))

    print("THE RESULT IF THE IMAGES ARE SAME")
    get_calculate_the_metrics(img_ori, img_ori)
    print("\n\n\n")

    print("The Result for original VGG19 for each metrics")
    get_calculate_the_metrics(img_ori, img_gen_original_vgg19)
    print("\n\n\n")

    print("The Result for trained VGG19 for each metrics")
    get_calculate_the_metrics(img_ori, img_gen_trained_vgg19, weight_path=model_weight_path)




