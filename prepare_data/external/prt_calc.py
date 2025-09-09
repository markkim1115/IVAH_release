import os
from scipy import io as sio
from prepare_data.external.prt.prt_util import computePRT


# Precomputed Radiance Transfer (PRT)
# Need "Pyembree" for quick operation
"""
# Pyembree: https://github.com/scopatz/pyembree
# Trimesh Advanced Installation Guide: https://trimsh.org/install.html
# Windows Pyembree install: https://github.com/scopatz/pyembree/issues/14#issuecomment-433406567
"""


def PRT_calc(opt, mesh, obj_path=None):
    """
    :param opt: Engine Option
    :param mesh: Trimesh Object
    :param obj_path: Directory information for saving
    :return: PRT Matrix
    """
    n = opt.prt_n
    order = opt.order
    # reserved_scale = float(load_json('./params/dataset_scale.json')[opt.dataset])
    # mesh.vertices *= reserved_scale

    PRT, face = computePRT(opt, mesh, n, order)

    if obj_path is not None:
        obj_dirname = os.path.dirname(obj_path)
        save_dir = os.path.join(obj_dirname, 'prt_data.mat')#./THuman2.0/0020/result/prt_data.mat
        if opt.verbose:
            print(f"PRT count: {n}")
            print(f"PRT order: {order}")
            print(f"save PRT matrix: {save_dir}")

        sio.savemat(
            save_dir,
            {'bounce0': PRT, 'face': face},
            do_compression=True
        )

    return {'bounce0': PRT, 'face': face}
