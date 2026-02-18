import torch
from torch import Tensor
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import numpy as np
import lpips
import utils as u

from typing import Tuple, List
def structural_similarity_index(image_original: Tensor, image_generated: Tensor, pixel_range: float, normalize: bool=True) -> Tensor:
    
    if normalize:
        image_original, image_generated = convert_to_device(image_original, image_generated)
        image_original = image_original.permute(2, 0, 1) # from HxWxC -> CxHxW
        image_generated = image_generated.permute(2, 0, 1)

        image_original = image_original.unsqueeze(dim=0)
        image_generated = image_generated.unsqueeze(dim=0)


        ssim_metric = StructuralSimilarityIndexMeasure(data_range=pixel_range)
        return ssim_metric(image_original, image_generated)

    else:
        raise ValueError("Too lazy brother, just normalized the image plszzzz")
def peak_signal_to_noise_ratio(image_original: Tensor, image_generated: Tensor, pixel_range: float, normalized: bool=True ) -> Tensor:
    if normalized:
        image_original, image_generated = convert_to_device(image_original, image_generated)
        mse = mean_squared_error(image_original, image_generated)
        psnr = 10 * torch.log10((pixel_range ** 2)/mse)
        return psnr
    else:
        raise ValueError("I am too lazy doing this, please just normalized your image")
    
def learned_perceptual_image_patch_similarity(image_original: Tensor, image_generated: Tensor, weight_path: str=None,
                                              normalize: bool = True, device: str = "cpu"):
    if weight_path:
        loss_fn_vgg_19 = lpips.LPIPS(net="vgg",model_path=weight_path)
    else:
        loss_fn_vgg_19 = lpips.LPIPS(net="vgg")
    

    image_original, image_generated = convert_to_device(image_original, image_generated, device=device)
    image_original = image_original.permute(2, 0, 1)
    image_generated = image_generated.permute(2, 0, 1)

    image_original = image_original.unsqueeze(dim=0)
    image_generated = image_generated.unsqueeze(dim=0)

    lp = loss_fn_vgg_19(image_original, image_generated, normalize=normalize)
    return lp

def mean_squared_error(image_original: Tensor, image_generated: Tensor, device: str="cpu") -> Tensor:
    "measured the mean squared error of the given generated content image and style image"
    image_original, image_generated = convert_to_device(image_original, image_generated, device)
    
    mse = torch.nn.MSELoss(reduction="mean")(image_original, image_generated)
    return mse


def convert_to_device(image_original: Tensor, image_generated: Tensor, device: str = "cpu") -> Tuple[Tensor, Tensor]:
    image_original = image_original.to(device)
    image_generated = image_generated.to(device)
    return image_original, image_generated


def test_lpips_same_images_with_weight():
    path_to_image_1: str = "./content_images/kk.jpg"
    saved_model_path: str = "./saved_model/batik_vgg19_features.pth"

    img_1: Tensor = torch.from_numpy(u.load_image(path_to_image_1, (400,400)))

    lpips = learned_perceptual_image_patch_similarity(img_1, img_1, weight_path=saved_model_path)
    print("#TEST 1: USING SAME IMAGES with LPIPS")

    print(f" LPIPS: {lpips.item():.2f} Expected: = 0.00")
    assert lpips.item() == 0.00
    print("Same images, pass")

def test_lpips_same_images():
    path_to_image_1: str = "./content_images/kk.jpg"

    img_1: Tensor = torch.from_numpy(u.load_image(path_to_image_1, (400,400)))

    lpips = learned_perceptual_image_patch_similarity(img_1, img_1)
    print("#TEST 1: USING SAME IMAGES with LPIPS")

    print(f" LPIPS: {lpips.item():.2f} Expected: = 0.00")
    assert lpips.item() == 0.00
    print("Same images, pass")



def test_lpips_different_images():
    
    path_to_image_1: str = "./content_images/kk.jpg"
    path_to_image_2: str = "./style_images/memes.jpg"

    img_1: Tensor = torch.from_numpy(u.load_image(path_to_image_1, (400, 400)))
    img_2: Tensor = torch.from_numpy(u.load_image(path_to_image_2, (400,400)))

    lpips = learned_perceptual_image_patch_similarity(img_1, img_2)
    print("#TEST 1: USING DIFFERENT IMAGES with LPIPS")

    print(f" LPIPS: {lpips.item():.2f} Expected: != 0.00")
    assert lpips.item() != 0.00
    print("different images, pass")


def test_structural_similarity_index_same_images():
    path_to_image_1: str = "./content_images/kk.jpg"

    img_1: Tensor = torch.from_numpy(u.load_image(path_to_image_1, (400,800)))

    ssi = structural_similarity_index(img_1, img_1, pixel_range=1.0)
    print("#TEST 1: USING SAME IMAGES with SSI")

    print(f" SSI: {ssi.item():.2f} Expected: = 1.00")
    assert round(ssi.item(),2) == 1.00
    print("Same image, pass")


def test_structural_similarity_index_different_images():
    path_to_image_1: str = "./content_images/kk.jpg"
    path_to_image_2: str = "./style_images/memes.jpg"

    img_1: Tensor = torch.from_numpy(u.load_image(path_to_image_1, (400, 400)))
    img_2: Tensor = torch.from_numpy(u.load_image(path_to_image_2, (400,400)))

    ssi = structural_similarity_index(img_1, img_2, pixel_range=1.0)
    print("#TEST 1: USING DIFFERENT IMAGES with SSI")

    print(f" SSI: {ssi.item():.2f} Expected: != 1.00")
    assert round(ssi.item(),2) != 1.00
    print("Different image, pass")

def test_psnr_same_images():

    path_to_image_1: str = "./content_images/kk.jpg"

    img_1: Tensor = torch.from_numpy(u.load_image(path_to_image_1, 400))

    mse = mean_squared_error(img_1, img_1)
    psnr = peak_signal_to_noise_ratio(img_1, img_1, 1.0)

    print("#TEST 1: USING SAME IMAGES")

    print(f" MSE: {mse.item():.4f} Expected: = 0")
    print(f" PSNR: {psnr.item():.4f} Expected: infinite")
    assert mse.item() == 0
    assert torch.isinf(psnr)

    print("Same image, pass")


def test_psnr_different_images():
    path_to_image_1: str = "./content_images/kk.jpg"
    path_to_image_2: str = "./style_images/memes.jpg"

    img_1: Tensor = torch.from_numpy(u.load_image(path_to_image_1, (400, 400)))
    img_2: Tensor = torch.from_numpy(u.load_image(path_to_image_2, (400,400)))

    mse = mean_squared_error(img_1, img_2)
    psnr = peak_signal_to_noise_ratio(img_1, img_2, 1.0)

    print("#TEST 2: USING DIFFERENT IMAGES")
    print(f" MSE: {mse.item():.4f} Expected: != 0")
    print(f" PSNR: {psnr.item():.4f} Expected: finite")

    assert mse.item() != 0
    assert torch.isfinite(psnr)
    print("Different image, pass")
if __name__ == "__main__":
#    test_lpips_same_images()
#    test_lpips_different_images()
#
#    test_psnr_same_images()
#    test_psnr_different_images()
#
#    test_structural_similarity_index_same_images()
#    test_structural_similarity_index_different_images()
    test_lpips_same_images_with_weight()