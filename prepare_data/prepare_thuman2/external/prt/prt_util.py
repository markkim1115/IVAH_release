import os

import numpy as np
import math

import argparse
import scipy.io as sio

from tqdm import tqdm
import parmap


def factratio(N, D):
    if N >= D:
        prod = 1.0
        for i in range(D+1, N+1):
            prod *= i
        return prod
    else:
        prod = 1.0
        for i in range(N+1, D+1):
            prod *= i
        return 1.0 / prod


def KVal(M, L):
    return math.sqrt(((2 * L + 1) / (4 * math.pi)) * (factratio(L - M, L + M)))


def AssociatedLegendre(M, L, x):
    if M < 0 or M > L or np.max(np.abs(x)) > 1.0:
        return np.zeros_like(x)
    
    pmm = np.ones_like(x)
    if M > 0:
        somx2 = np.sqrt((1.0 + x) * (1.0 - x))
        fact = 1.0
        for i in range(1, M+1):
            pmm = -pmm * fact * somx2
            fact = fact + 2
    
    if L == M:
        return pmm
    else:
        pmmp1 = x * (2 * M + 1) * pmm
        if L == M+1:
            return pmmp1
        else:
            pll = np.zeros_like(x)
            for i in range(M+2, L+1):
                pll = (x * (2 * i - 1) * pmmp1 - (i + M - 1) * pmm) / (i - M)
                pmm = pmmp1
                pmmp1 = pll
            return pll


def SphericalHarmonic(M, L, theta, phi):
    if M > 0:
        return math.sqrt(2.0) * KVal(M, L) * np.cos(M * phi) * AssociatedLegendre(M, L, np.cos(theta))
    elif M < 0:
        return math.sqrt(2.0) * KVal(-M, L) * np.sin(-M * phi) * AssociatedLegendre(-M, L, np.cos(theta))
    else:
        return KVal(0, L) * AssociatedLegendre(0, L, np.cos(theta))


def save_obj(mesh_path, verts):
    file = open(mesh_path, 'w')    
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    file.close()


def sampleSphericalDirections(n):
    xv = np.random.rand(n,n)
    yv = np.random.rand(n,n)
    theta = np.arccos(1-2 * xv)
    phi = 2.0 * math.pi * yv

    phi = phi.reshape(-1)
    theta = theta.reshape(-1)

    vx = -np.sin(theta) * np.cos(phi)
    vy = -np.sin(theta) * np.sin(phi)
    vz = np.cos(theta)
    return np.stack([vx, vy, vz], 1), phi, theta


def getSHCoeffs(order, phi, theta):
    shs = []
    for n in range(0, order+1):
        for m in range(-n,n+1):
            s = SphericalHarmonic(m, n, theta, phi)
            shs.append(s)
    
    return np.stack(shs, 1)


def multiPRT(input_arg):
    i, n, mesh, vectors_orig, SH_orig, n_v, origins, normals = input_arg

    SH = np.repeat(SH_orig[None, (i * n):((i + 1) * n)], n_v, axis=0).reshape(-1, SH_orig.shape[1])
    vectors = np.repeat(vectors_orig[None, (i * n): ((i + 1) * n)], n_v, axis=0).reshape(-1, 3)

    dots = (vectors * normals).sum(1)
    front = (dots > 0.0)

    delta = 1e-3 * min(mesh.bounding_box.extents)
    hits = mesh.ray.intersects_any(origins + delta * normals, vectors)
    nohits = np.logical_and(front, np.logical_not(hits))

    PRT = (nohits.astype(np.float64) * dots)[:, None] * SH
    PRT = PRT.reshape(-1, n, SH.shape[1]).sum(1)

    return PRT


def computePRT(opt, mesh, n, order):
    """
    :param opt: Engine Option
    :param mesh: Trimesh Object
    :param n:
    :param order:
    :return: PRT Matrix
    """
    vectors_orig, phi, theta = sampleSphericalDirections(n)
    SH_orig = getSHCoeffs(order, phi, theta)

    w = 4.0 * math.pi / (n*n)

    origins = mesh.vertices
    normals = mesh.vertex_normals
    n_v = origins.shape[0]

    origins = np.repeat(origins[:, None], n, axis=1).reshape(-1, 3)
    normals = np.repeat(normals[:, None], n, axis=1).reshape(-1, 3)

    if opt.num_processes > 1:
        # Python Multi-Processing
        # * Issues
        # https://github.com/mikedh/trimesh/issues/171
        # https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        iter_args = [(i, n, mesh, vectors_orig, SH_orig, n_v, origins, normals) for i in range(n)]
        PRT_list = parmap.map(multiPRT, iter_args, pm_pbar=True, pm_processes=opt.num_processes)
    else:
        PRT_list = []
        for i in tqdm(range(n)):
            PRT_list.append(multiPRT((i, n, mesh, vectors_orig, SH_orig, n_v, origins, normals)))

    PRT = w * sum(PRT_list)
    del PRT_list

    return PRT, mesh.faces


def testPRT(obj_path, n=40):
    dir_path, obj_fname = os.path.split(obj_path)

    if dir_path[-1] == '/':
        dir_path = dir_path[:-1]
    os.makedirs(os.path.join(dir_path, 'bounce'), exist_ok=True)

    PRT, F = computePRT(obj_path, n, 2)
    sio.savemat(
        os.path.join(dir_path, 'bounce', 'prt_data.mat'), 
        {'bounce0': PRT, 'face': F}, 
        do_compression=True)
    # np.save(os.path.join(dir_path, 'bounce', 'bounce0.npy'), PRT)
    # np.save(os.path.join(dir_path, 'bounce', 'face.npy'), F)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/home/shunsuke/Downloads/rp_dennis_posed_004_OBJ')
    parser.add_argument('-n', '--n_sample', type=int, default=40, help='squared root of number of sampling. the higher, the more accurate, but slower')
    args = parser.parse_args()

    testPRT(args.input, args.n_sample)
