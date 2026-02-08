import torch
from torch import Tensor
from typing import List, Set

def compute_content_loss(
        NN_feature_maps_list: List[Tensor],
        NN_feature_maps_index: int,
        original_content_representation: Tensor
):
    # Low content loss means generated image has similar structure to the original
    current_generated_content_representation: Tensor = NN_feature_maps_list[NN_feature_maps_index].squeeze(axis=0) # returns (1, C, H, W) -> (C, H, W)
    return torch.nn.MSELoss(reduction="mean")(original_content_representation, current_generated_content_representation) # Averages the squarted differences across all elements

def compute_style_loss(NN_feature_maps_list: List[Tensor],
                       NN_feature_maps_indices: List[int],
                       original_style_representation: Tensor
                       ):
    current_generated_style_representation: List[Tensor] = []
    style_loss: float = 0.0

    
    for indx, f_map in enumerate(NN_feature_maps_list):
        if indx in NN_feature_maps_indices: #For each layers that is specified by the NN_feature_maps_indices, calculate its layer gram_matrix
            current_generated_style_representation.append(gram_matrix(f_map))

    for original_gram, generated_gram in zip(original_style_representation, current_generated_style_representation):
        style_loss += torch.nn.MSELoss(reduction="sum")(original_gram, generated_gram)

    return style_loss / len(NN_feature_maps_list)
    

def compute_tv_loss(input_image: Tensor):
    numerator: Tensor = torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:]))
    denumerator: Tensor = torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
    return numerator / denumerator

def compute_total_loss( content_loss: Tensor, style_loss: Tensor,
                        TV_loss: Tensor, TV_weight: float,
                        content_weight: float, style_weight: float,
                        ):
    """
        Returns: 
        Weighted total Loss
    """

    return (content_weight * content_loss) + (style_weight * style_loss) + (TV_weight * TV_loss)

def gram_matrix(input_tensor: Tensor, normalize: bool = True):
    (b, c, h, w) = input_tensor.size()
    flatenned_features = input_tensor.view(b, c, h*w) # (b, c, features)
    flatenned_features_transposed = flatenned_features.transpose(1,2) # (b, features, c)
    gram_mat: Tensor = flatenned_features.bmm(flatenned_features_transposed) # (b, c, features) matmul (b, features, c) -> (b, c, c)
    if normalize:
        gram_mat /= c * h * w # divide by the number total of elements in the feature map
    return gram_mat

