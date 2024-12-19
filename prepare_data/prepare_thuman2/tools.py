import os
import json
import numpy as np
import argparse

def str2bool(sentence):
    if isinstance(sentence, bool):
        return sentence
    elif sentence.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif sentence.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def mesh_normalize(opt, mesh, obj_path=None):
    overwrite = opt.norm_overwrite #Whether to overwrite the normalized mesh in the input mesh file ./THuman2.0/0020/0020.obj
    verbose = opt.verbose #Whether to print detailed information about ongoing tasks

    min_xyz = np.min(mesh.vertices, axis=0, keepdims=True)#To get the maximum value for each axis, specify the axis number as the axis argument
    max_xyz = np.max(mesh.vertices, axis=0, keepdims=True)#To keep the dimensional shape, set the keepdims argument to True

    # mesh.vertices = mesh.vertices - (min_xyz + max_xyz) * 0.5#difference value from midpoint

    norm_trans = - (min_xyz + max_xyz) * 0.5

    scale_inv = np.max(max_xyz - min_xyz)#The largest value among the thicknesses of each of the 3 axes (xyz)
    # scale = 1. / scale_inv * (0.75 + np.random.rand() * 0.15)
    # scale = 1. / scale_inv # Scale Factor
    
    if True: # raw mesh rendering
        scale = 1.0
        norm_trans = 0

    mesh.vertices *= scale # difference value from midpoint * Scale Factor
    

    if obj_path is not None:  # save normalized mesh
        dirname = os.path.join(obj_path)
        os.makedirs(dirname, exist_ok=True)
        if not overwrite:
            filename = os.path.splitext(os.path.basename(obj_path))[0] + '_norm.obj'
            obj_path = os.path.join(dirname, filename) # ./THuman2.0/0020/result/0020_norm.obj
        else:
            print(f"original file is overwritten.")

        if verbose:
            print(f"Scale Factor: {scale}")
            print(f"normalized file save directory: {obj_path}")

        # trimesh.base.export_mesh(mesh, obj_path)
        mesh.export(obj_path) # save normalized mesh ./THuman2.0/0020/result/0020_norm.obj
        # np.savetxt(os.path.join(dirname, 'scale.txt'), np.array([scale]))

    return mesh, os.path.basename(obj_path), scale, norm_trans

def mesh_normalize_by_smpl(opt, mesh, smpl_scale, smpl_transl, obj_path):
    mesh.vertices = (mesh.vertices - smpl_transl) / smpl_scale # make scale into SMPL model scale
    mesh.vertices += smpl_transl / smpl_scale

    min_xyz = np.min(mesh.vertices, axis=0, keepdims=True) # To get the maximum value for each axis, specify the axis number as the axis argument
    max_xyz = np.max(mesh.vertices, axis=0, keepdims=True) # To keep the dimensional shape, set the keepdims argument to True

    norm_trans = - (min_xyz + max_xyz) * 0.5
    # mesh.vertices = mesh.vertices - (min_xyz + max_xyz) * 0.5 # difference value from midpoint

    scale_inv = np.max(max_xyz - min_xyz)#The largest value among the thicknesses of each of the 3 axes (xyz)
    scale = 1. / scale_inv # Scale Factor
    # mesh.vertices *= scale # difference value from midpoint * Scale Factor

    if obj_path is not None:  # save normalized mesh
        dirname = os.path.join(obj_path)
        os.makedirs(dirname, exist_ok=True)
        subject = obj_path.split('/')[-1]
        filename = f'{subject}' + '_norm.obj'
        obj_path = os.path.join(dirname, filename)
        print(f"SMPL_Scale Factor: {smpl_scale}")
        print(f"Scale Factor: {scale}")
        print(f"normalized file save directory: {obj_path}")

        mesh.export(obj_path) 
        # np.savetxt(os.path.join(dirname, 'scale.txt'), np.array([scale]))

    return mesh, os.path.basename(obj_path), smpl_scale, scale, norm_trans

def save_json(obj: dict, name):
    for key, value in obj.items():
        if isinstance(value, np.ndarray):
            obj[key] = value.tolist()
    with open(name, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(param_dir):
    with open(param_dir) as f:
        param = json.load(f)

    return param