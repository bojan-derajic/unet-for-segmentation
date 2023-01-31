import numpy as np

def draw_segmentation_mask(image, mask, alpha):
    mask_R = np.zeros_like(image)
    mask_R[:, :, 0] = mask[:, :, 0]

    mask_G = np.zeros_like(image)
    mask_G[:, :, 1] = mask[:, :, 1]

    image = np.uint8((1 - alpha*np.repeat(np.expand_dims(mask[:, :, 0], axis=2), repeats=3, axis=2))*image + alpha*mask_R*255)
    image = np.uint8((1 - alpha*np.repeat(np.expand_dims(mask[:, :, 1], axis=2), repeats=3, axis=2))*image + alpha*mask_G*255)

    return image