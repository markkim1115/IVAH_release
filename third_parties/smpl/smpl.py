import torch
import numpy as np
from third_parties.smpl.smplx import SMPL
import os
import pickle
MODEL_DIR = 'third_parties/smpl/models'

def load_pickle_of_template():
    # modified From SHERF official code
    model_path = os.path.join(MODEL_DIR, 'SMPL_NEUTRAL.pkl')
    with open(model_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

def load_smpl_model(device='cpu'):
    smpl_model = SMPL(MODEL_DIR, gender='neutral').to(device)

    return smpl_model

def get_smpl_from_numpy_input(smpl_model, global_orient=None, body_pose=None, betas=None, transl=None, device='cpu'):
    if global_orient is None:
        global_orient = np.zeros((1,3), dtype=np.float32)
    if body_pose is None:
        body_pose = np.zeros((1,69), dtype=np.float32)
    if betas is None:
        betas = np.zeros((1,10), dtype=np.float32)
    smpl_model = smpl_model.to(device)
    global_orient = torch.from_numpy(global_orient).to(device).float()
    body_pose = torch.from_numpy(body_pose).to(device).float()
    betas = torch.from_numpy(betas).to(device).float()
    if transl is not None:
        transl = torch.from_numpy(transl).to(device).float()
    with torch.no_grad():
        pred_output = smpl_model(betas=betas,
                                    body_pose=body_pose,
                                    global_orient=global_orient,
                                    transl=transl,
                                    pose2rot=True
                                    )
    return pred_output