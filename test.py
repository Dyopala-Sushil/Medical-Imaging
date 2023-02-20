import torch
from model import getModel
import utils
import numpy as np
import sys

if __name__ == "__main__":
    path2best_model = sys.argv[1]
    path2data = sys.argv[2  ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = utils.load_model(path2best_model, device)

    processed_data = utils.preprocess_data(path2data)
    # Adding batch dimension
    processed_data = processed_data[None, ...]

    predicted_seg_masks = utils.get_prediction_masks(processed_data, model, device)

    post_processed_data = utils.post_process_prediction(predicted_seg_masks[0])

    # Generating background and foreground from predicted segmentation mask
    bg = post_processed_data[0,...]
    fg = post_processed_data[1,...]

    # Reshaping to (H, W, D) dimensions only
    fg = fg.cpu().detach().numpy()
    bg = bg.cpu().detach().numpy()
    image_vol = processed_data.cpu().detach().numpy()
    image_vol = np.squeeze(image_vol, axis=(0,1)) 

    seg_img_vol = utils.overlay_mask(image_vol, bg, fg)

    utils.save_segmented_result(seg_img_vol)
