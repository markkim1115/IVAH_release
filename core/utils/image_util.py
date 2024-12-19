import os
import shutil

from termcolor import colored
from PIL import Image
import torch
import numpy as np

def image_exhibit_set(cfg, gt=False, alphamap=False):
    if gt:
        cfg.show_truth = True
    if alphamap:
        cfg.show_alpha = True

def load_image(path, to_rgb=True):
    img = Image.open(path)
    return img.convert('RGB') if to_rgb else img


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def to_8b_image(image):
    return (255.* np.clip(image, 0., 1.)).astype(np.uint8)


def to_3ch_image(image):
    if len(image.shape) == 2:
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        assert image.shape[2] == 1
        return np.concatenate([image, image, image], axis=-1)
    else:
        print(f"to_3ch_image: Unsupported Shapes: {len(image.shape)}")
        return image


def to_8b3ch_image(image):
    return to_3ch_image(to_8b_image(image))


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))
    
def output_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None):
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))
    else:
        truth = None

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image  = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image

def output_to_alphamap_tensor(img_width, img_height, ray_mask, alpha):
    
    width = img_width
    height = img_height

    alpha_map = torch.zeros((height * width)).float().to(alpha.device)
    alpha_map[ray_mask] = alpha

    return alpha_map.reshape((height, width))

def tile_images(images, imgs_per_row=4):
    rows = []
    row = []
    imgs_per_row = min(len(images), imgs_per_row)
    for i in range(len(images)):
        row.append(images[i])
        if len(row) == imgs_per_row:
            rows.append(np.concatenate(row, axis=1))
            row = []
    if len(rows) > 2 and len(rows[-1]) != len(rows[-2]):
        rows.pop()
    imgout = np.concatenate(rows, axis=0)
    return imgout

     
class ImageWriter():
    def __init__(self, output_dir, exp_name, clean_up=False):
        self.image_dir = os.path.join(output_dir, exp_name)

        print("The rendering is saved in " + \
              colored(self.image_dir, 'cyan'))
        
        # remove image dir if it exists
        if os.path.exists(self.image_dir) and clean_up == True:
            shutil.rmtree(self.image_dir)
        
        os.makedirs(self.image_dir, exist_ok=True)
        self.frame_idx = -1

    def append(self, image, img_name=None):
        self.frame_idx += 1
        if img_name is None:
            img_name = f"{self.frame_idx:06d}"
        save_image(image, f'{self.image_dir}/{img_name}.png')
        return self.frame_idx, img_name

    def finalize(self):
        pass
