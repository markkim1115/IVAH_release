import torch
import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
from PIL import Image
from core.utils.uv_utils import UVMapGenerator
from UVTextureConverter import UVConverter
from UVTextureConverter import Atlas2Normal
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
import skimage
import os
import argparse

class RGB2DensePose:

    def __init__(self) -> None:
        pass

    '''
    Computes densepose to all images in input_folder/ and saves .png and .pkl
    '''
    def folder2densepose(self, root, subject, camera_name):

        # gets parent directory of input folder (WARNING: input_folder should not end with '/')
        parent_directory = root
        input_folder = os.path.join(root, subject, 'img_upsampled', camera_name)
        print ("Image dir " + input_folder)
        print("Parent: " + parent_directory)

        # creates densepose/ directory
        densepose_image_directory = os.path.join(parent_directory, subject, 'densepose', camera_name)
        isExist = os.path.exists(densepose_image_directory)
        if not isExist:
            os.makedirs(densepose_image_directory)

        # creates densepose_pkl/ directory
        densepose_pkl_directory = os.path.join(parent_directory, subject, 'densepose_pkl', camera_name)
        isExist = os.path.exists(densepose_pkl_directory)
        if not isExist:
            os.makedirs(densepose_pkl_directory)
        files = sorted(os.listdir(input_folder))

        output_pkl = os.path.join(densepose_pkl_directory, "output.pkl")
        os.system("python apply_net.py dump configs/densepose_rcnn_R_101_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl " + input_folder + " --output " + output_pkl + " -v")

        with open(output_pkl, 'rb') as f:
            data = torch.load(f)
        
        for i in range(len(data)):
            image_name = os.path.basename(data[i]['file_name'])[:-4]
            self.data2densepose(data[i], os.path.join(densepose_image_directory, f"{image_name}_densepose.png"))
        
    def data2densepose(self, data, output_image):
        
        img_path = data['file_name']
        img     = Image.open(img_path)
        img_w ,img_h = img.size

        i       = data['pred_densepose'][0].labels.cpu().numpy()
        uv      = data['pred_densepose'][0].uv.cpu().numpy()
        iuv     = np.stack((uv[1,:,:], uv[0,:,:], 256 - i))
        iuv     = np.transpose(iuv, (1,2,0))
        iuv_img = Image.fromarray(np.uint8(iuv*255),"RGB")
        
        #iuv_img.show() #It shows only the croped person

        box     = data["pred_boxes_XYXY"][0]
        box[2]  = box[2]-box[0]
        box[3]  = box[3]-box[1]
        x,y,w,h = [int(v) for v in box]
        
        bg      = np.zeros((img_h,img_w,3))
        bg[y:y+h,x:x+w,:] = iuv
        bg_img  = Image.fromarray(np.uint8(bg*255),"RGB")

        bg_img.save(output_image)

    def img2densepose(self, input_image, output_pkl, output_image):
        
        os.system("python apply_net.py dump configs/densepose_rcnn_R_101_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl " + input_image + " --output " + output_pkl + " -v")
        
        img          = Image.open(input_image)
        img_w ,img_h = img.size

        # loads .pkl with dense pose data
        with open(output_pkl, 'rb') as f:
            data = torch.load(f)
        
        i       = data[0]['pred_densepose'][0].labels.cpu().numpy()
        uv      = data[0]['pred_densepose'][0].uv.cpu().numpy()
        iuv     = np.stack((uv[1,:,:], uv[0,:,:], 256 - i))
        iuv     = np.transpose(iuv, (1,2,0))
        iuv_img = Image.fromarray(np.uint8(iuv*255),"RGB")
        
        #iuv_img.show() #It shows only the croped person

        box     = data[0]["pred_boxes_XYXY"][0]
        box[2]  = box[2]-box[0]
        box[3]  = box[3]-box[1]
        x,y,w,h = [int(v) for v in box]
        
        bg      = np.zeros((img_h,img_w,3))
        bg[y:y+h,x:x+w,:] = iuv
        bg_img  = Image.fromarray(np.uint8(bg*255),"RGB")

        bg_img.save(output_image)

class RGB2Texture:

    def __init__(self, dataset_root_path) -> None:
        self.dataset_root_path = dataset_root_path
        
    def apply_mask_to_iuv_images(self, subject, camera_name):
        dataset_iuv_path = os.path.join(self.dataset_root_path, subject, 'densepose', camera_name)
        dataset_mask_path = os.path.join(self.dataset_root_path, subject, 'mask_upsampled', camera_name)

        if not os.path.exists(dataset_iuv_path):
            print("ERROR: ", dataset_iuv_path, " does not exist")
            return

        if not os.path.exists(dataset_mask_path):
            print("ERROR: ", dataset_mask_path, " does not exist")
            return

        # output path for masked iuv
        output_iuv_masked_folder = "densepose_masked"
        output_iuv_masked_folder_path = os.path.join(self.dataset_root_path, subject, output_iuv_masked_folder, camera_name)
        isExist = os.path.exists(output_iuv_masked_folder_path)
        if not isExist:
            os.makedirs(output_iuv_masked_folder_path)

        files = sorted(os.listdir(dataset_iuv_path))

        num_images = 0

        # reads all the images and stores full paths in list
        for path in files:

            num_images += 1
            current_iuv_path = os.path.join(dataset_iuv_path, path)
            current_mask_path = path.replace("_densepose.png", ".png")
            current_mask_path = os.path.join(dataset_mask_path, current_mask_path)

            if os.path.isfile(current_iuv_path):
                if os.path.isfile(current_mask_path):

                    with Image.open(current_iuv_path) as im_iuv:
                        with Image.open(current_mask_path) as im_mask:

                            print('\nSegmenting image ', num_images, '/', len(files))
                            #print('Loading      ', current_iuv_path)
                            #print('Loading      ', current_mask_path)

                            iuv_w, iuv_h = im_iuv.size
                            mask_w, mask_h = im_mask.size

                            if (iuv_w == mask_w) and (iuv_h == mask_h):
                                
                                threshold = 250
                                im_mask = im_mask.point(lambda x: 255 if x > threshold else 0)
                                blank = im_iuv.point(lambda _: 0)
                                masked_iuv_image = Image.composite(im_iuv, blank, im_mask)
                                
                                print('Writing image ', os.path.join(output_iuv_masked_folder_path, path))
                                masked_iuv_image.save(os.path.join(output_iuv_masked_folder_path, path), "PNG")
                            else:

                                print('Discarding images because densepose and RGB do not match. Probably densepose failed?')
                                
            
                else:
                    print(current_mask_path, 'does not exist')
            else:
                print(current_iuv_path, 'does not exist')


    def generate_uv_texture(self, subject, camera_name):
    
        # paths
        dataset_image_path = os.path.join(self.dataset_root_path, subject, 'img_upsampled', camera_name)
        dataset_mask_path = os.path.join(self.dataset_root_path, subject, 'mask_upsampled', camera_name)
        dataset_iuv_path = os.path.join(self.dataset_root_path, subject, 'densepose_masked', camera_name)
        
        # output path for UV textures
        output_textures_folder = "uv_textures"
        output_textures_folder_path = os.path.join(self.dataset_root_path, subject, output_textures_folder, camera_name)
        isExist = os.path.exists(output_textures_folder_path)
        if not isExist:
            os.makedirs(output_textures_folder_path)

        # output path for debug figure
        output_debug_folder = "debug"
        output_debug_folder_path = os.path.join(self.dataset_root_path, subject, output_debug_folder, camera_name)
        isExist = os.path.exists(output_debug_folder_path)
        if not isExist:
            os.makedirs(output_debug_folder_path)

        # list to store files
        images_file_paths = []
        masks_file_paths = []
        images_iuv_paths = []

        num_images = 0

        # extract list of image files
        files = os.listdir(dataset_image_path)

        # size (in pixels) of each part in texture
        # WARNING: for best results, update this value depending on the input image size
        parts_size = 120 

        # reads all the images and stores full paths in list
        for path in files:
            num_images += 1
            current_image_path = os.path.join(dataset_image_path, path)
            current_mask_path = os.path.join(dataset_mask_path, path)
            current_iuv_path = os.path.join(dataset_iuv_path, path.replace(".png", "_densepose.png"))

            # check if both image and iuv exists
            if os.path.isfile(current_image_path):
                if os.path.isfile(current_iuv_path):
                    images_file_paths.append(current_image_path)
                    masks_file_paths.append(current_mask_path)
                    images_iuv_paths.append(current_iuv_path)
                    # print('\nProcessing image ', num_images, '/', len(files))
                    # print(current_image_path)
                    # print(current_iuv_path)

                else:
                    print(current_iuv_path, ' does not exist')
            else:
                print(current_image_path, ' does not exist')

        num_images = 0

        # sorts filenames alphabetically
        images_iuv_paths.sort()
        images_file_paths.sort()
        masks_file_paths.sort()

        images_iuv_paths_filtered = images_iuv_paths.copy()
        images_file_paths_filtered = images_file_paths.copy()
        masks_file_paths_filtered = masks_file_paths.copy()

        num_images = 0

        for current_image_path, current_mask_path, current_iuv_path in zip(images_file_paths_filtered, masks_file_paths_filtered, images_iuv_paths_filtered):
            
            num_images += 1
            
            print('\nComputing UV texture ', num_images, '/', len(images_file_paths_filtered))
            # create texture by densepose in atlas style
            # tex_trans: (24, 200, 200, 3), mask_trans: (24, 200, 200)
            tex_trans, mask_trans = self.create_texture(current_image_path, current_iuv_path,
                                                        parts_size=parts_size, concat=False, msk=current_mask_path)

            # convert from atlas to normal
            texture_size = 512 

            converter = Atlas2Normal(atlas_size=parts_size, normal_size=texture_size)
            normal_tex, normal_ex = converter.convert((tex_trans*255).astype('int'), mask=mask_trans)

            # shows result
            fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(13,4))
            ax1.imshow(normal_tex)
            
            ax2.set_title(os.path.basename(current_image_path), fontsize=8)
            ax2.imshow(mpimg.imread(current_image_path))
            
            ax3.set_title(os.path.basename(current_iuv_path), fontsize=8)
            ax3.imshow(mpimg.imread(current_iuv_path))
            
            # plt.show()

            # file names for debug and output texture            
            output_debug_filename = os.path.basename(current_image_path).replace(".jpg", "_debug.png")
            output_debug_file_path = os.path.join(output_debug_folder_path, output_debug_filename)

            output_texture_filename = os.path.basename(current_image_path).replace(".jpg", "_texture.png")
            output_textures_file_path = os.path.join(output_textures_folder_path, output_texture_filename)

            # save debug image
            print('Saving         ', output_debug_file_path)
            plt.savefig(output_debug_file_path, dpi=150)
            plt.close(fig) 

            # save output uv texture
            normal_tex = (normal_tex * 255).round().astype(np.uint8)
            im = Image.fromarray(normal_tex, 'RGB')
            im.save(output_textures_file_path)
            print('Saving         ', output_textures_file_path)
            
            previous_tex = normal_tex

    def create_smpl_from_images(self, im, iuv, msk=None, img_size=200):
        i_id, u_id, v_id = 2, 1, 0
        parts_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        parts_num = len(parts_list)
        
        # generate parts
        if isinstance(im, str) and isinstance(iuv, str):
            im = Image.open(im)
            iuv = Image.open(iuv)
            if msk is not None:
                msk = Image.open(msk)
        elif isinstance(im, type(Image)) and isinstance(iuv, type(Image)):
            im = im
            iuv = iuv
            if msk is not None:
                msk = msk
        elif isinstance(im, np.ndarray) and isinstance(iuv, np.ndarray):
            im = Image.fromarray(np.uint8(im * 255))
            iuv = Image.fromarray(np.uint8(iuv * 255))
            if msk is not None:
                msk = Image.fromarray(np.uint8(msk * 255))
        else:
            raise ValueError('im and iuv must be str or PIL.Image or np.ndarray.')
        
        # Masking with segmentation mask
        if msk is not None:
            msk_np = np.array(msk).copy()
            msk_eroded = cv2.erode(msk_np, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
            msk_eroded = Image.fromarray(msk_eroded)

            im_eroded = Image.composite(im, Image.new('RGB', iuv.size, (0, 0, 0)), msk_eroded)
            erode_inpaint_mask = np.logical_xor(msk_eroded, np.array(msk))
            erode_inpaint_mask = Image.fromarray(erode_inpaint_mask)
            im_inpainted = cv2.inpaint(np.array(im_eroded), np.array(erode_inpaint_mask).astype(np.uint8), 3, cv2.INPAINT_TELEA)
            im = Image.fromarray(im_inpainted)
            
        im = (np.array(im) / 255).transpose(2, 1, 0)
        iuv = (np.array(iuv)).transpose(2, 1, 0)

        texture = np.zeros((parts_num, 3, img_size, img_size))
        mask = np.zeros((parts_num, img_size, img_size))
        for j, parts_id in enumerate(parts_list):
            im_gen = np.zeros((3, img_size, img_size))
            im_gen[0][(iuv[v_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int),
                      (iuv[u_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int)] = im[0][iuv[i_id] == parts_id]
            im_gen[1][(iuv[v_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int),
                      (iuv[u_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int)] = im[1][iuv[i_id] == parts_id]
            im_gen[2][(iuv[v_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int),
                      (iuv[u_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int)] = im[2][iuv[i_id] == parts_id]
            texture[j] = im_gen[:, ::-1, :]

            mask[j][(iuv[v_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int),
                    (iuv[u_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int)] = 1
            mask[j] = mask[j][::-1, :]

        return texture, mask

    def create_texture(self, im, iuv, parts_size=200, concat=True, msk=None):
        tex, mask = self.create_smpl_from_images(im, iuv, img_size=parts_size, msk=msk) # tex: (24, 3, W, H), mask: (24, W, H)
        

        tex_trans = np.zeros((24, parts_size, parts_size, 3))
        mask_trans = np.zeros((24, parts_size, parts_size))
        for i in range(tex.shape[0]):
            tex_trans[i] = tex[i].transpose(2, 1, 0)
            mask_trans[i] = mask[i].transpose(1, 0)
        if concat:
            return UVConverter.concat_atlas_tex(tex_trans), UVConverter.concat_atlas_tex(mask_trans)
        else:
            return tex_trans, mask_trans
        
    def compute_mask_of_partial_uv_textures(self, subject, camera_name):

        # folder where the uv textures are. Ideally, these are textures generated with masked images 
        # (e.g, the IUV images where segmented using masks computed on the rendered images)
        uv_textures_path = os.path.join(self.dataset_root_path, subject, 'uv_textures', camera_name)
        
        files = os.listdir(uv_textures_path)

        # output path for UV masks
        output_masks_folder = "uv_textures_masks"
        output_masks_folder_path = os.path.join(self.dataset_root_path, subject, output_masks_folder, camera_name)
        isExist = os.path.exists(output_masks_folder_path)
        if not isExist:
            os.makedirs(output_masks_folder_path)

        num_images = 0

        # reads all the images and computes UV mask
        for image_filename in files:
            num_images += 1
            print('\nComputing UV mask ', num_images, '/', len(files))

            current_image = skimage.io.imread(os.path.join(uv_textures_path, image_filename))

            mask = (current_image[:, :, 0] < 2) & (current_image[:, :, 1] < 2) & (current_image[:, :, 1] < 2)
            mask = ~mask
            mask_filename = image_filename.replace(".png", "_mask.png")
            print('Writing image ', os.path.join(output_masks_folder_path, mask_filename))
            skimage.io.imsave(os.path.join(output_masks_folder_path, mask_filename), skimage.img_as_ubyte(mask))

def to_3ch_uint8(img):
    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)
    if len(img.shape) == 2:
        return np.stack([img, img, img], axis=-1)
    return img

def pre_process_cur_tex(tex, texmask, texmask_dilation_kernel=5, texmask_dilation_iter=1, remove_contour=True):
     
    if (tex.dtype != np.uint8):
        tex = to_3ch_uint8(tex)
    if (texmask.dtype != np.uint8):
        texmask = texmask.astype(np.uint8)

    dilated_mask = cv2.dilate(texmask, cv2.getStructuringElement(cv2.MORPH_RECT, (texmask_dilation_kernel, texmask_dilation_kernel)), iterations=texmask_dilation_iter).astype(bool)
    and_mask = np.logical_and(texmask, dilated_mask.astype(bool))
    inpaint_mask = np.logical_xor(and_mask, dilated_mask).astype(bool)
    
    inpainted = cv2.inpaint(src=tex, inpaintMask=inpaint_mask.astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    if remove_contour:
        gray = cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY)
        
        contours_vis = np.zeros_like(gray)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cv2.drawContours(contours_vis, contours, i, (255,255,255), 1)
        
        dilated_contour = cv2.dilate(contours_vis, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1) # Remove contour color artifacts with small dilation
        contours_mask = dilated_contour.astype(bool)
        
        inpainted[contours_mask] = 0
    return inpainted

class MergeUVs():
    def __init__(self) -> None:
        SMPL_UV_mask_path = 'third_parties/smpl/models/smpl_uv_20200910.png'
        smpl_uv_mask = Image.open(SMPL_UV_mask_path).resize((512,512))
        smpl_uv_mask = np.array(smpl_uv_mask)
        smpl_uv_mask = smpl_uv_mask[:,:,-1] > 0
        self.smpl_uv_mask = smpl_uv_mask.astype(bool)
        
        uv_hand_template_map = 'third_parties/smpl/models/smpl_uv_20200910_hand.png'
        uv_hand = Image.open(uv_hand_template_map).resize((512,512))
        uv_hand = np.array(uv_hand)
        uv_hand_mask = uv_hand[:,:,-1] > 0
        self.uv_hand_mask = uv_hand_mask

        uv_face_template_map = 'third_parties/smpl/models/smpl_uv_20200910_face.png'
        uv_face = Image.open(uv_face_template_map).resize((512,512))
        uv_face = np.array(uv_face)
        uv_face_mask = uv_face[:,:,-1] > 0
        self.uv_face_mask = uv_face_mask

        self.vertex_colors = np.zeros((6890, 3))
        self.h = 512
        self.w = 512
        self.gnr = UVMapGenerator(self.h)
        
        self.texcoords = self.gnr.texcoords * np.array([[self.h - 1, self.w - 1]])
        self.vt_to_v_index_set = np.array([self.gnr.vt_to_v[i] for i in range(self.texcoords.shape[0])])
    
    def inpaint_face(self, cur_tex, cur_msk):
        cur_msk = cur_msk.astype(bool)
        self.uv_face_mask = self.uv_face_mask.astype(bool)
        partial_face_mask = np.logical_and(cur_msk, self.uv_face_mask)
        Image.fromarray(to_3ch_uint8(partial_face_mask.astype(np.uint8)*255)).show()
        partial_face_mask = cv2.dilate(partial_face_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=3)
        Image.fromarray(to_3ch_uint8(partial_face_mask.astype(np.uint8)*255)).show()

        inpainted_face = cv2.inpaint(cur_tex, partial_face_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        Image.fromarray(to_3ch_uint8(inpainted_face.astype(np.uint8)*255)).show()
        cur_tex[partial_face_mask] = inpainted_face[partial_face_mask]
        return cur_tex
    
    def merge_uv_maps_on_smpl_vertex(self, root, subject, camera_names):
        # Set base hand color
        pose_idx=0
        uv_textures_path = os.path.join(root, subject, 'uv_textures', camera_names[0])
        if not os.path.exists(uv_textures_path):
            print("ERROR: ", uv_textures_path, " does not exist")
            exit(0)
        uv_masks_path = os.path.join(root, subject, 'uv_textures_masks', camera_names[0])
        if not os.path.exists(uv_masks_path):
            print("ERROR: ", uv_masks_path, " does not exist")
            exit(0)

        uv_tex_frame_name = f"{pose_idx:04d}.png"
        uv_mask_frame_name = f"{pose_idx:04d}_mask.png"

        cur_tex = Image.open(os.path.join(uv_textures_path, uv_tex_frame_name))
        cur_msk = Image.open(os.path.join(uv_masks_path, uv_mask_frame_name))
        
        cur_tex = np.array(cur_tex)
        cur_msk = np.array(cur_msk).astype(bool)
        
        hand_only = cur_tex.copy()
        hand_only = hand_only * self.uv_hand_mask[:,:,None].astype(np.uint8)
        self.hand_color = np.max(hand_only, axis=(0,1)).astype(np.uint8)
        
        for pose_idx in range(18,19):
            for camera_name in camera_names:
                cur_vert_colors = np.zeros((6890, 3))
                uv_textures_path = os.path.join(root, subject, 'uv_textures', camera_name)
                if not os.path.exists(uv_textures_path):
                    print("ERROR: ", uv_textures_path, " does not exist")
                    exit(0)
                uv_masks_path = os.path.join(root, subject, 'uv_textures_masks', camera_name)
                if not os.path.exists(uv_masks_path):
                    print("ERROR: ", uv_masks_path, " does not exist")
                    exit(0)

                uv_tex_frame_name = f"{pose_idx:04d}.png"
                uv_mask_frame_name = f"{pose_idx:04d}_mask.png"

                cur_tex = Image.open(os.path.join(uv_textures_path, uv_tex_frame_name))
                cur_msk = Image.open(os.path.join(uv_masks_path, uv_mask_frame_name))
                
                cur_tex = np.array(cur_tex)
                cur_msk = np.array(cur_msk).astype(bool)
                
                cur_tex_inpainted_1 = pre_process_cur_tex(cur_tex, cur_msk, texmask_dilation_kernel=5, texmask_dilation_iter=1, remove_contour=False)
                cur_tex_inpainted_2 = self.inpaint_face(cur_tex_inpainted_1, cur_msk)
                cur_tex_inpainted = cur_tex_inpainted_2.copy()
                cur_tex_inpainted[self.uv_hand_mask] = self.hand_color
                Image.fromarray(cur_tex_inpainted).save(f"uv_merge_{pose_idx:03d}_{camera_name}.png")
        
    def merge_uv_maps_on_uvmap(self, root, subject, camera_names):

        tex = np.zeros((512, 512, 3))
        msk = np.zeros((512, 512))
        camera_names = [camera_names[0], camera_names[9], camera_names[18], camera_names[27]]
        for pose_idx in range(0, 1):
            for camera_name in camera_names:
                uv_textures_path = os.path.join(root, subject, 'uv_textures', camera_name)
                if not os.path.exists(uv_textures_path):
                    print("ERROR: ", uv_textures_path, " does not exist")
                    exit(0)
                uv_masks_path = os.path.join(root, subject, 'uv_textures_masks', camera_name)
                if not os.path.exists(uv_masks_path):
                    print("ERROR: ", uv_masks_path, " does not exist")
                    exit(0)
                uv_textures_merged_path = os.path.join(root, subject, 'uv_textures_merged', camera_name)
                isExist = os.path.exists(uv_textures_merged_path)
                if not isExist:
                    os.makedirs(uv_textures_merged_path)
                uv_tex_frame_name = f"{pose_idx:04d}.png"
                uv_mask_frame_name = f"{pose_idx:04d}_mask.png"

                cur_tex = Image.open(os.path.join(uv_textures_path, uv_tex_frame_name))
                cur_msk = Image.open(os.path.join(uv_masks_path, uv_mask_frame_name))
                
                cur_tex = np.array(cur_tex)
                cur_msk = np.array(cur_msk).astype(bool)
                
                cur_tex_inpainted = pre_process_cur_tex(cur_tex, cur_msk, texmask_dilation_kernel=9, texmask_dilation_iter=2, remove_contour=True)
                inpainted_msk = (cur_tex_inpainted[:,:,0] > 2) & (cur_tex_inpainted[:,:,1] > 2) & (cur_tex_inpainted[:,:,2] > 2)
                merge_msk = np.logical_or(msk.astype(bool), inpainted_msk.astype(bool)).astype(bool)

                overlab_msk = np.logical_and(msk.astype(bool), inpainted_msk.astype(bool)).astype(bool)
                diff_msk = np.logical_xor(overlab_msk.astype(bool), inpainted_msk.astype(bool)).astype(bool)

                tex[diff_msk] = cur_tex_inpainted[diff_msk]
                vis = np.concatenate((to_3ch_uint8(msk*255), to_3ch_uint8(cur_msk*255), to_3ch_uint8(inpainted_msk*255), to_3ch_uint8(overlab_msk), to_3ch_uint8(diff_msk*255), cur_tex, cur_tex_inpainted, tex), axis=1).astype(np.uint8)
                Image.fromarray(vis).save(f"uv_merge_{pose_idx:03d}_{camera_name}.png")
                print(f"Saved uv_merge_{pose_idx:03d}_{camera_name}.png")
                msk = merge_msk.copy()
        
        tex = tex.astype(np.uint8)
        
        # Inpaint the final texture
        tex_mask_ = (tex[:, :, 0] <30) & (tex[:, :, 1] <30) & (tex[:, :, 1] <30)
        tex_mask_ = tex_mask_.astype(np.uint8)
        
        smpl_uv_mask = self.smpl_uv_mask.astype(bool)
        tex_mask = np.logical_and(tex_mask_, smpl_uv_mask).astype(np.uint8)
        
        
        final_tex = cv2.inpaint(tex, tex_mask, inpaintRadius=10, flags=cv2.INPAINT_NS)
        Image.fromarray(np.concatenate([tex, to_3ch_uint8(tex_mask_*255), to_3ch_uint8(tex_mask*255), final_tex], axis=1).astype(np.uint8)).show()
        Image.fromarray(final_tex).save(f"uv_merge_final.png")

arguments = argparse.ArgumentParser()
arguments.add_argument('--root', type=str, default='dataset/RenderPeople')
arguments.add_argument('--subject', type=str, default='subject_01')
arguments.add_argument('--detectron_path', type=str, default='/home/cv1/works/detectron2')
arguments.add_argument('--extract_densepose', action='store_true')
arguments.add_argument('--extract_partial_uv_textures', action='store_true')

args = arguments.parse_args()
DETECTRON_PATH = os.path.abspath(args.detectron_path)
ROOT_PATH  = os.path.abspath(args.root)
SUBJECT = args.subject

ALL_CAMERA_NUMBERS = np.arange(0, 36)
ALL_CAMERA_NAMES = [f'camera{i:04d}' for i in ALL_CAMERA_NUMBERS][:1]

extract_densepose = args.extract_densepose
extract_partial_uv_textures = args.extract_partial_uv_textures

sys.path.append(os.path.join(DETECTRON_PATH,"projects/DensePose")) 
cur_wd = os.getcwd()
if extract_densepose:
    for CAMERA_NAME in ALL_CAMERA_NAMES:
        image_folder = os.path.join(ROOT_PATH, SUBJECT, 'img_upsampled', CAMERA_NAME)    
        image_files = sorted(os.listdir(image_folder))
        num_poses = len(image_files)
        # Change path to your dir of detectron2/projects/DensePose

        os.chdir(os.path.join(DETECTRON_PATH,"projects/DensePose"))
        densepose = RGB2DensePose()
        densepose.folder2densepose(root=ROOT_PATH, subject=SUBJECT, camera_name=CAMERA_NAME)
        os.chdir(cur_wd)

if extract_partial_uv_textures:
    for cam_idx, CAMERA_NAME in enumerate(ALL_CAMERA_NAMES):
        densepose_dir = os.path.join(ROOT_PATH, SUBJECT, 'densepose', CAMERA_NAME)
        if not os.path.exists(densepose_dir):
            print("ERROR: ", densepose_dir, " does not exist")
            exit(0)
        # Change path to your dir of detectron2/projects/DensePose
        texture = RGB2Texture(ROOT_PATH)
        texture.apply_mask_to_iuv_images(SUBJECT, CAMERA_NAME)
        texture.generate_uv_texture(SUBJECT, CAMERA_NAME)
        texture.compute_mask_of_partial_uv_textures(SUBJECT, CAMERA_NAME)