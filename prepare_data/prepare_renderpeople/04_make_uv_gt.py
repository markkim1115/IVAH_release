import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
import numpy as np
import cv2
import PIL.Image as Image
import pickle
from time import time
from numpy.linalg import solve
from skimage.io import imsave
from tqdm import tqdm
from core.utils.camera_util import apply_global_tfm_to_camera
from core.utils.vis_util import create_3d_figure, mesh_object, draw_2D_joints
from third_parties.smpl.smpl import get_smpl_from_numpy_input, load_smpl_model

def to_3ch_uint8(img):
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    if len(img.shape) < 3:
        ch_3 = np.stack([img, img, img], axis=-1).astype(np.uint8)
        if img.max() < 1:
            ch_3 = (ch_3 * 255).astype(np.uint8)
    
    return ch_3

class UVMapGenerator():
    def __init__(self, h=256):
        self.h = h
        self.w = w = h

        processed_pickle_file = 'third_parties/smpl/models/smpl_uv.pkl'
        if not os.path.exists(processed_pickle_file):
            obj_file = 'third_parties/smpl/models/smpl_uv.obj'
            self._parse_obj(obj_file, 'third_parties/smpl/models/smpl_uv.pkl')
        else:
            with open(processed_pickle_file, 'rb') as f:
                tmp = pickle.load(f)
            for k in tmp.keys():
                setattr(self, k, tmp[k])

        self.cache_data_root = cache_data_root = 'third_parties/smpl/models'
        self.bc_pickle = f'{cache_data_root}/barycentric_h{h:04d}_w{w:04d}.pkl'

        ### Load (or calcluate) barycentric info
        if not os.path.exists(self.bc_pickle):
            bary_id, bary_weights, edge_dict = self._calc_bary_info(h, w, self.vt_faces.shape[0])
        else:
            with open(self.bc_pickle, 'rb') as rf:
                bary_dict = pickle.load(rf)
            bary_id = bary_dict['face_id']
            bary_weights = bary_dict['bary_weights']
            edge_dict = bary_dict['edge_dict']

        self.bary_id = bary_id
        self.bary_weights = bary_weights
        self.edge_dict = edge_dict

    def _parse_obj(self, obj_file, cache_file):
        with open(obj_file, 'r') as fin:
            lines = [l 
                for l in fin.readlines()
                if len(l.split()) > 0
                and not l.startswith('#')
            ]
        
        # Load all vertices (v) and texcoords (vt)
        vertices = []
        texcoords = []
        
        for line in lines:
            lsp = line.split()
            if lsp[0] == 'v':
                x = float(lsp[1])
                y = float(lsp[2])
                z = float(lsp[3])
                vertices.append((x, y, z))
            elif lsp[0] == 'vt':
                u = float(lsp[1])
                v = float(lsp[2])
                texcoords.append((1-v, u))
                
        # Stack these into an array
        self.vertices = np.vstack(vertices).astype(np.float32)
        self.texcoords = np.vstack(texcoords).astype(np.float32)
        
        # Load face data. All lines are of the form:
        # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
        #
        # Store the texcoord faces and a mapping from texcoord faces
        # to vertex faces
        vt_faces = []
        self.vt_to_v = {} # texcoord to vertex mapping
        self.v_to_vt = [None] * self.vertices.shape[0] # vertex to tex coord mapping
        for i in range(self.vertices.shape[0]):
            self.v_to_vt[i] = set()
        
        for line in lines:
            vs = line.split()
            
            if vs[0] == 'f':
                v0 = int(vs[1].split('/')[0]) - 1
                v1 = int(vs[2].split('/')[0]) - 1
                v2 = int(vs[3].split('/')[0]) - 1
                vt0 = int(vs[1].split('/')[1]) - 1
                vt1 = int(vs[2].split('/')[1]) - 1
                vt2 = int(vs[3].split('/')[1]) - 1
                vt_faces.append((vt0, vt1, vt2))
                self.vt_to_v[vt0] = v0
                self.vt_to_v[vt1] = v1
                self.vt_to_v[vt2] = v2
                self.v_to_vt[v0].add(vt0)
                self.v_to_vt[v1].add(vt1)
                self.v_to_vt[v2].add(vt2)
                
        self.vt_faces = np.vstack(vt_faces)
        tmp_dict = {
            'vertices': self.vertices,
            'texcoords': self.texcoords,
            'vt_faces': self.vt_faces,
            'vt_to_v': self.vt_to_v,
            'v_to_vt': self.v_to_vt
        }
        # with open(cache_file, 'wb') as w:
        #     pickle.dump(tmp_dict, w)

    '''
    __: for a given uv vertice position,
    return the berycentric information for all pixels
    
    Parameters:
    ------------------------------------------------------------
    h, w: image size
    faces: [F * 3], triangle pieces represented with UV vertices.
    uvs: [N * 2] UV coordinates on texture map, scaled with h & w
    
    Output: 
    ------------------------------------------------------------
    bary_dict: A dictionary containing three items, saved as pickle.
    @ face_id: [H*W] int tensor where f[i,j] represents
        which face the pixel [i,j] is in
        if [i,j] doesn't belong to any face, f[i,j] = -1
    @ bary_weights: [H*W*3] float tensor of barycentric coordinates 
        if f[i,j] == -1, then w[i,j] = [0,0,0]
    @ edge_dict: {(u,v):n} dict where (u,v) indicates the pixel to 
        dilate, with n being non-zero neighbors within 8-neighborhood
        
        
    Algorithm:
    ------------------------------------------------------------
    The barycentric coordinates are obtained by 
    solving the following linear equation:
    
            [ x1 x2 x3 ][w1]   [x]
            [ y1 y2 y3 ][w2] = [y]
            [ 1  1  1  ][w3]   [1]
    
    Note: This algorithm is not the fastest but can be batchlized.
    It could take 8~10 minutes on a regular PC for 300*300 maps. 
    Luckily, for each experiment the bary_info only need to be 
    computed once, so I just stick to the current implementation.
    '''
    
    def _calc_bary_info(self, h, w, F):
        s = time()
        face_id = np.zeros((h, w), dtype=int)
        bary_weights = np.zeros((h, w, 3), dtype=np.float32)
        
        uvs = self.texcoords * np.array([[self.h - 1, self.w - 1]])
        grids = np.ones((F, 3), dtype=np.float32)
        anchors = np.concatenate((
            uvs[self.vt_faces].transpose(0,2,1),
            np.ones((F, 1, 3), dtype=uvs.dtype)
        ), axis=1) # [F * 3 * 3] F = 13776, num_vertices = 6890, num_vt = 7576
        
        _loop = tqdm(np.arange(h*w), ncols=80)
        for i in _loop:
            r = i // w
            c = i % w
            grids[:, 0] = r
            grids[:, 1] = c
            
            weights = solve(anchors, grids) # not enough accuracy?
            inside = np.logical_and.reduce(weights.T > 1e-10)
            index = np.where(inside == True)[0]
            
            if 0 == index.size:
                face_id[r,c] = -1  # just assign random id with all zero weights.
            #elif index.size > 1:
            #    print('bad %d' %i)
            else:
                face_id[r,c] = index[0]
                bary_weights[r,c] = weights[index[0]]

        # calculate counter pixels for UV_map dilation
        _mask = np.where(face_id == -1, 0, 1)
        edge_dict = {}
        _loop = _loop = tqdm(np.arange((h-2)*(w-2)), ncols=80)
        for l in _loop:
            i = l // (w-2) + 1 
            j = l % (w-2) + 1
            _neighbor = np.array([
                _mask[i-1, j], _mask[i-1, j+1],
                _mask[i, j+1], _mask[i+1, j+1],
                _mask[i+1, j], _mask[i+1, j-1],
                _mask[i, j-1], _mask[i-1, j-1],
            ])
            
            if _mask[i,j] == 0 and _neighbor.min() != _neighbor.max():
                edge_dict[(i, j)] = np.count_nonzero(_neighbor)
                    
            
        print('Calculating finished. Time elapsed: {}s'.format(time()-s))
        
        bary_dict = {
            'face_id': face_id,
            'bary_weights': bary_weights,
            'edge_dict': edge_dict
        }
        
        with open(self.bc_pickle, 'wb') as wf:
            pickle.dump(bary_dict, wf)
            
        return face_id, bary_weights, edge_dict
        
    '''
        _dilate: perform dilate_like operation for initial
        UV map, to avoid out-of-region error when resampling
    '''
    def _dilate(self, UV_map, pixels=1):
        _UV_map = UV_map.copy()
        for k, v in self.edge_dict.items():
            i, j = k[0], k[1]
            _UV_map[i, j] = np.sum(np.array([
                UV_map[i-1, j], UV_map[i-1, j+1],
                UV_map[i, j+1], UV_map[i+1, j+1],
                UV_map[i+1, j], UV_map[i+1, j-1],
                UV_map[i, j-1], UV_map[i-1, j-1],
            ]), axis=0) / v
    
        return _UV_map
    
    #####################
    # UV Map Generation #
    #####################
    '''
    UV_interp: barycentric interpolation from given
    rgb values at non-integer UV vertices.
    
    Parameters:
    ------------------------------------------------------------
    rgbs: [N * 3] rgb colors at given uv vetices.
    
    Output: 
    ------------------------------------------------------------
    UV_map: colored texture map with the same size as im
    '''    
    def UV_interp(self, rgbs):
        face_num = self.vt_faces.shape[0]
        vt_num = self.texcoords.shape[0]
        assert(vt_num == rgbs.shape[0])
        
        uvs = self.texcoords * np.array([[self.h - 1, self.w - 1]])
        
        #print(np.max(rgbs), np.min(rgbs))
        triangle_rgbs = rgbs[self.vt_faces][self.bary_id]
        bw = self.bary_weights[:,:,np.newaxis,:]
        #print(triangle_rgbs.shape, bw.shape)
        im = np.matmul(bw, triangle_rgbs).squeeze(axis=2)
        
        '''
        for i in range(height):
            for j in range(width):
                t0 = faces[fid[i,j]][0]
                t1 = faces[fid[i,j]][1]
                t2 = faces[fid[i,j]][2]
                im[i,j] = (w[i,j,0] * rgbs[t0] + w[i,j,1] * rgbs[t1] + w[i,j,2] * rgbs[t2])
        '''
        
        #print(im.shape, np.max(im), np.min(im))
        im = np.minimum(np.maximum(im, 0.), 1.)
        return im
        
    '''
    get_UV_map: create UV position map from aligned mesh coordinates
    Parameters:
    ------------------------------------------------------------
    verts: [V * 3], aligned mesh coordinates.
    
    Output: 
    ------------------------------------------------------------
    UV_map: [H * W * 3] Interpolated UV map.
    colored_verts: [H * W * 3] Scatter plot of colorized UV vertices
    '''
    def get_UV_map(self, vertex_colors=None, dilate=True):
        if vertex_colors is None:
            # normalize all to [0,1]
            _min = np.amin(verts, axis=0, keepdims=True)
            _max = np.amax(verts, axis=0, keepdims=True)
            verts = (verts - _min) / (_max - _min)
            verts_backup = verts.copy()
            vertex_colors = verts
        else:
            verts_backup = vertex_colors.copy()
            if vertex_colors.max() > 1:
                vertex_colors = vertex_colors / 255.
            
        vt_to_v_index = np.array([
            self.vt_to_v[i] for i in range(self.texcoords.shape[0])
        ])
        rgbs = vertex_colors[vt_to_v_index]
        
        uv_map = self.UV_interp(rgbs)
        if dilate:
            uv_map = self._dilate(uv_map)
        return uv_map, verts_backup
    
    def render_point_cloud(self, img_name=None, verts=None, rgbs=None, eps=1e-8):
        if verts is None:
            verts = self.vertices
        if rgbs is None:
            #print('Warning: rgb not specified, use normalized 3d coords instead...')
            v_min = np.amin(verts, axis=0, keepdims=True)
            v_max = np.amax(verts, axis=0, keepdims=True)
            rgbs = (verts - v_min) / np.maximum(eps, v_max - v_min)
        
        vt_id = [self.vt_to_v[i] for i in range(self.texcoords.shape[0])]
        img = np.zeros((self.h, self.w, 3), dtype=rgbs.dtype)
        uvs = (self.texcoords * np.array([[self.h - 1, self.w - 1]])).astype(int)
        
        img[uvs[:, 0], uvs[:, 1]] = rgbs[vt_id]
        
        if img_name is not None:
            imsave(img_name, img)
            
        return img, verts, rgbs
    
    def write_ply(self, ply_name, verts, rgbs=None, eps=1e-8):
        if rgbs is None:
            #print('Warning: rgb not specified, use normalized 3d coords instead...')
            v_min = np.amin(verts, axis=0, keepdims=True)
            v_max = np.amax(verts, axis=0, keepdims=True)
            rgbs = (verts - v_min) / np.maximum(eps, v_max - v_min)
        if rgbs.max() < 1.001:
            rgbs = (rgbs * 255.).astype(np.uint8)
        
        with open(ply_name, 'w') as f:
            # headers
            f.writelines([
                'ply\n'
                'format ascii 1.0\n',
                'element vertex {}\n'.format(verts.shape[0]),
                'property float x\n',
                'property float y\n',
                'property float z\n',
                'property uchar red\n',
                'property uchar green\n',
                'property uchar blue\n',
                'end_header\n',
                ]
            )
            
            for i in range(verts.shape[0]):
                str = '{:10.6f} {:10.6f} {:10.6f} {:d} {:d} {:d}\n'\
                    .format(verts[i,0], verts[i,1], verts[i,2],
                        rgbs[i,0], rgbs[i,1], rgbs[i,2])
                f.write(str)
                
        return verts, rgbs
    
    def complete_uv_map_from_sparse_colored_smpl_vertex(self, sparse_vertex_colors):
        # img, verts, rgbs = gnr.render_point_cloud('render_point_cloud.png', None, sparse_vertex_colors)
        uv, _ = gnr.get_UV_map(sparse_vertex_colors, dilate=False)
        uv = (uv * 255).astype(np.uint8)
        
        return uv

def inpaint_densepose_uv_map(tex, texmask, texmask_dilation_kernel=11, texmask_dilation_iter=4, remove_contour=True, rm_contour_kernel=13, rm_contour_iter=3):
     
    if (tex.dtype != np.uint8):
        tex = to_3ch_uint8(tex)
    if (texmask.dtype != np.uint8):
        texmask = texmask.astype(np.uint8)

    # Close holes in the mask
    _tex_mask = cv2.dilate(texmask, cv2.getStructuringElement(cv2.MORPH_RECT, (texmask_dilation_kernel, texmask_dilation_kernel)), iterations=texmask_dilation_iter)
    _tex_mask = cv2.erode(_tex_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (texmask_dilation_kernel, texmask_dilation_kernel)), iterations=texmask_dilation_iter).astype(bool)
    
    # Inpaint
    inpaint_mask = np.logical_xor(texmask, _tex_mask)
    inpainted = cv2.inpaint(src=tex, inpaintMask=inpaint_mask.astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Remove contour color artifacts
    if remove_contour:
        gray = cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY)
        
        contours_vis = np.zeros_like(gray)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cv2.drawContours(contours_vis, contours, i, (255,255,255), 1)
        
        dilated_contour = cv2.dilate(contours_vis, cv2.getStructuringElement(cv2.MORPH_RECT, (rm_contour_kernel, rm_contour_kernel)), iterations=rm_contour_iter) # Remove contour color artifacts with small dilation
        contours_mask = dilated_contour < 2
        
        inpainted = inpainted * contours_mask[...,None]
    return inpainted

def get_contour(mask, dilation_kernel_size=13, dilation_iter=2):
    contour_img = np.zeros_like(mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv2.drawContours(contour_img, contours, i, (255,255,255), 1)
    dilated_contour = cv2.dilate(contour_img, cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_kernel_size, dilation_kernel_size)), iterations=dilation_iter)

    return dilated_contour

uv_face_template_map = 'third_parties/smpl/models/smpl_uv_20200910_face.png'
uv_face = Image.open(uv_face_template_map).resize((512,512))
uv_face = np.array(uv_face)
uv_face_mask = uv_face[:,:,-1] > 0
uv_face_mask = uv_face_mask.astype(np.uint8)

uv_body_mask = Image.open('third_parties/smpl/models/smpl_uv_20200910_body.png').resize((512,512))
uv_body_mask = np.array(uv_body_mask)
uv_body_mask = uv_body_mask[:,:,-1] > 0
uv_body_mask = uv_body_mask.astype(np.uint8)

uv_hand_template_map = 'third_parties/smpl/models/smpl_uv_20200910_hand.png'
uv_hand = Image.open(uv_hand_template_map).resize((512,512))
uv_hand = np.array(uv_hand)
uv_hand_mask = uv_hand[:,:,-1] > 0
uv_hand_mask = uv_hand_mask

gnr = UVMapGenerator(512)
bary_weights = gnr.bary_weights
bary_id = gnr.bary_id
# edge_dict = gnr.edge_dict

device = 'cpu'
smpl_model = load_smpl_model('cpu')
smpl_faces = smpl_model.faces

# import matplotlib.pyplot as plt
# cmap = plt.get_cmap('jet')
# colors = [cmap(i) for i in np.linspace(0, 1, len(smpl_face))]
# colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
# colors = np.array(colors, dtype=np.uint8)
# colors[-1] = colors[-1] * 0

# bary_id_vis = np.zeros((512, 512, 3), dtype=np.uint8).reshape(-1, 3)
# bary_id_vis=colors[bary_id]
# bary_id_vis = bary_id_vis.reshape(512, 512, 3)
# Image.fromarray(bary_id_vis).show()
# pdb.set_trace()

data_root = 'dataset/RenderPeople'

subjects = [x for x in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, x))]

for subject_idx in range(len(subjects)):
    subject = subjects[subject_idx]
    print(f'Processing {subject} ...')
    subject_root = os.path.join(data_root, subject)

    face_fix_targets = [1,140,141,143,145,154,169,272,273,274,485,605]
    subject_num = int(subject.split('-')[0].split('_')[-1])

    if os.path.exists(os.path.join(subject_root, 'uvmap_gt.png')) and (subject_num not in face_fix_targets):
        continue

    smpl_path = os.path.join(data_root, subject, f'outputs_re_fitting/refit_smpl_2nd.npz')
    
    smpl_data = np.load(smpl_path, allow_pickle=True)['smpl'][()]
    
    global_orient = smpl_data['global_orient'] # (p,3)
    body_pose = smpl_data['body_pose'] # (p,69)
    betas = smpl_data['betas'] # (1,10)
    transl = smpl_data['transl'] # (p,3)
    smpl_out = get_smpl_from_numpy_input(smpl_model, np.zeros_like(global_orient), body_pose, np.tile(betas, (global_orient.shape[0],1)), np.zeros_like(transl), device)
    smpl_verts_all = smpl_out.vertices.detach().cpu().numpy()
    smpl_joints_all = smpl_out.joints.detach().cpu().numpy()
    smpl_root_joint = smpl_joints_all[0,[0]]
    
    tpose_smpl = get_smpl_from_numpy_input(smpl_model, None, None, None, None, device)
    tpose_verts = tpose_smpl.vertices.detach().cpu().numpy()[0]

    camera_infos_path = os.path.join(subject_root, 'cameras.pkl')
    with open(camera_infos_path ,'rb') as f:
        camera_infos = pickle.load(f)

    camera_view_names = np.array(sorted([x for x in camera_infos.keys()]))[[0,9,18,27]]
    
    smpl_visibility_root = os.path.join(subject_root, 'smpl_vertex_visibility_mask.pkl')
    with open(smpl_visibility_root, 'rb') as f:
        smpl_visibility_dataset = pickle.load(f)
    
    vertex_color_buffer = np.zeros((6890, 3), dtype=np.float32)
    colored_verts_mask = np.zeros((6890), dtype=int)
    for pose_idx in range(smpl_verts_all.shape[0]):
        for view_idx, view in enumerate(camera_view_names):
            camera_info = camera_infos[view]
            camera_extrinsic = camera_info['extrinsics']
            camera_intrinsic = camera_info['intrinsics']
            image_path = os.path.join(subject_root, 'img', f'camera{view}', f'{pose_idx:04d}.jpg')
            mask_path = os.path.join(subject_root, 'mask', f'camera{view}', f'{pose_idx:04d}.png')

            img_pil = Image.open(image_path)
            img = np.array(img_pil)
            mask_pil = Image.open(mask_path)
            mask = np.array(mask_pil).astype(np.uint8)
            msk_eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1) # Valid mask 영역을 줄인다.
            im_eroded = img * (msk_eroded[:,:,None]/255.) # Edge Artifact 제거
            dilated_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11,11)), iterations=1) # Valid mask 영역을 늘린다.
            
            erode_inpaint_mask = np.logical_xor(msk_eroded.astype(bool), dilated_mask.astype(bool)) # SMPL vertex가 fit하지 않을 때를 대비하여 더 큰 영역을 inpaint한다.
            img = cv2.inpaint(im_eroded.astype(np.uint8), erode_inpaint_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
            
            h,w = img.shape[:2]

            global_orient_matrix = cv2.Rodrigues(global_orient[pose_idx])[0]
            raw_transl = transl[pose_idx]
            transl_ = smpl_root_joint - np.matmul(smpl_root_joint, global_orient_matrix.T) + raw_transl
            transl_ = transl_[0]

            K = camera_intrinsic
            E = apply_global_tfm_to_camera(camera_extrinsic, global_orient[pose_idx], transl_)
            R = E[:3, :3]
            T = E[:3, 3]
            smpl_visibility = smpl_visibility_dataset[view][f'{pose_idx:04d}']
            
            vertex_world = smpl_verts_all[pose_idx]
            vertex_cam = vertex_world@R.T + T
            vertex_cam = vertex_cam@K.T
            vertex2d = vertex_cam[:,:2] / vertex_cam[:,2:]

            visible_vertices_2d = vertex2d[smpl_visibility]
            visible_vertices_2d = visible_vertices_2d.astype(int)
            # vis = draw_2D_joints(img.astype(np.uint8), visible_vertices_2d)
            # if os.path.exists('vertex_renderings') == False:
            #     os.makedirs('vertex_renderings')
            # Image.fromarray(vis).save("vertex_renderings/RP_{}_VIEW_{}_POSE_{}.png".format(subject, view, pose_idx))
            
            if visible_vertices_2d[:,0].max() > (img.shape[1]-1) or visible_vertices_2d[:,1].max() > (img.shape[0]-1):
                print("Warning: Visible vertices are out of image boundary. Skip this view.")
                continue
            else:
                visible_vertex_colors = img[visible_vertices_2d[:,1], visible_vertices_2d[:,0]]

            cur_vertex_color = np.zeros((6890, 3), dtype=np.float32)
            cur_vertex_color[smpl_visibility] = visible_vertex_colors

            cur_vertex_mask = np.zeros((6890), dtype=bool)
            cur_vertex_mask[smpl_visibility] = True
            target_mask = np.logical_and(colored_verts_mask, cur_vertex_mask)
            target_mask = np.logical_xor(target_mask, cur_vertex_mask)
            
            cur_vertex_color = cur_vertex_color * target_mask[:,None].astype(int)

            vertex_color_buffer[target_mask] = cur_vertex_color[target_mask]
            
            colored_verts_mask = np.logical_or(colored_verts_mask, cur_vertex_mask)
            print(f'{subject} | POSE {pose_idx+1}/{smpl_verts_all.shape[0]} | VIEW {view_idx+1}/{camera_view_names.shape[0]} done.')
    # objs = []
    # objs += [mesh_object(tpose_verts + np.array([[2,0,0]]), smpl_faces, vertex_color_buffer)]
    # create_3d_figure(objs, draw_grid=False).show()
                
    smpl_colors = vertex_color_buffer
    save_root = os.path.join(data_root, subject)
    save_path = os.path.join(save_root, 'uvmap_gt.png')

    uv = gnr.complete_uv_map_from_sparse_colored_smpl_vertex(smpl_colors)
    densepose_based_uv_fix_process = True
    if densepose_based_uv_fix_process:
        densepose_uv_textures_root = os.path.join(subject_root, 'uv_textures')
        densepose_uv_texture_path = os.path.join(densepose_uv_textures_root, 'camera0000', '0000.png')
        densepose_uv_texture = np.array(Image.open(densepose_uv_texture_path))

        densepose_uv_map_mask = ((densepose_uv_texture[...,0] > 2) & (densepose_uv_texture[...,1] > 2) & (densepose_uv_texture[...,2] > 2)).astype(np.uint8)
        
        densepose_uv_texture_map_face_mask = uv_face_mask * densepose_uv_map_mask
        densepose_uv_texture_face_masked = densepose_uv_texture * densepose_uv_texture_map_face_mask[...,None]
        inpainted_densepose_uv_map_face = inpaint_densepose_uv_map(densepose_uv_texture_face_masked, densepose_uv_texture_map_face_mask, rm_contour_kernel=13, rm_contour_iter=3)
        inpainted_densepose_uv_map_face_mask = ((inpainted_densepose_uv_map_face[...,0] > 2) & (inpainted_densepose_uv_map_face[...,1] > 2) & (inpainted_densepose_uv_map_face[...,2] > 2)).astype(np.uint8)

        contour_mask_face = get_contour(inpainted_densepose_uv_map_face_mask, dilation_kernel_size=13, dilation_iter=1)

        densepose_uv_map_body_mask = uv_body_mask * densepose_uv_map_mask
        densepose_uv_texture_body_masked = densepose_uv_texture * densepose_uv_map_body_mask[...,None]
        inpainted_densepose_uv_map_body = inpaint_densepose_uv_map(densepose_uv_texture_body_masked, densepose_uv_map_body_mask, texmask_dilation_kernel=3, texmask_dilation_iter=1, rm_contour_kernel=3, rm_contour_iter=1)
        inpainted_densepose_uv_map_body_mask = ((inpainted_densepose_uv_map_body[...,0] > 2) & (inpainted_densepose_uv_map_body[...,1] > 2) & (inpainted_densepose_uv_map_body[...,2] > 2)).astype(np.uint8)

        contour_mask_body = get_contour(inpainted_densepose_uv_map_body_mask, dilation_kernel_size=13, dilation_iter=1)

        smoothing_target_mask = np.logical_or(contour_mask_face, contour_mask_body)

    merged_uv = uv.copy()
    if densepose_based_uv_fix_process:
        if subject_num in face_fix_targets:
            merged_uv[inpainted_densepose_uv_map_face_mask == 1] = inpainted_densepose_uv_map_face[inpainted_densepose_uv_map_face_mask == 1]
        merged_uv[inpainted_densepose_uv_map_body_mask == 1] = inpainted_densepose_uv_map_body[inpainted_densepose_uv_map_body_mask == 1]
        merged_uv_raw = merged_uv.copy()
        # Gaussian smoothing inpainted part edge
        blured_uv = merged_uv.copy()
        blured_uv = cv2.GaussianBlur(blured_uv, (11,11), 0)

        merged_uv[smoothing_target_mask == 1] = blured_uv[smoothing_target_mask == 1]

    merged_uv = merged_uv.astype(np.uint8)
    if densepose_based_uv_fix_process:
        hand_only = merged_uv.copy()
        hand_only = hand_only * uv_hand_mask[...,None]
        hand_only = cv2.GaussianBlur(hand_only, (11,11), 0)
        merged_uv[uv_hand_mask] = hand_only[uv_hand_mask]
    
    # vis = np.concatenate([uv, densepose_uv_texture, 
    #                       uv_face[...,:3], 
    #                       to_3ch_uint8(densepose_uv_map_mask)*255, 
    #                       to_3ch_uint8(uv_face_mask)*255, 
    #                       densepose_uv_texture_face_masked, 
    #                       to_3ch_uint8(densepose_uv_texture_map_face_mask)*255, 
    #                       inpainted_densepose_uv_map_face,
    #                       inpainted_densepose_uv_map_body,
    #                       to_3ch_uint8(contour_mask_face),
    #                     to_3ch_uint8(contour_mask_body),
    #                     to_3ch_uint8(smoothing_target_mask),
    #                     merged_uv_raw,
    #                     blured_uv,
    #                     hand_only,
    #                       merged_uv], axis=1)
    # Image.fromarray(vis).show()
    Image.fromarray(merged_uv).save(save_path)
    print(f'{subject} UV map GT created, saved. {subject_idx+1}/{len(subjects)}')
print("Done!")