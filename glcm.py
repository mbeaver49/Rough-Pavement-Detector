import cv2
import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndimage

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def fast_glcm_cpu(
    img, vmin=0, vmax=255, levels=8, kernel_size=3, distance=1.0, angle=0.0
):
    mi, ma = vmin, vmax
    ks = kernel_size
    h, w = img.shape
    bins = np.linspace(mi, ma + 1, levels + 1)
    gl1 = np.digitize(img, bins) - 1
    dx = distance * np.cos(np.deg2rad(angle))
    dy = distance * np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    gl2 = cv2.warpAffine(
        gl1, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE
    )
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i, j, (gl1 == i) & (gl2 == j)] = 1
    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i, j] = cv2.filter2D(glcm[i, j], -1, kernel)
    return glcm.astype(np.float32)


def fast_glcm_gpu(
    img, vmin=0, vmax=255, levels=8, kernel_size=3, distance=1.0, angle=0.0
):
    mi, ma = vmin, vmax
    ks = kernel_size
    h, w = img.shape

    img_cp = cp.asarray(img, dtype=cp.float32)
    bins = cp.linspace(mi, ma + 1, levels + 1)
    gl1 = cp.digitize(img_cp, bins) - 1

    dx = distance * cp.cos(cp.deg2rad(angle))
    dy = distance * cp.sin(cp.deg2rad(-angle))

    gl2 = ndimage.shift(gl1, [-dy, -dx], order=0, mode="nearest")

    i_idx = cp.arange(levels).reshape(levels, 1, 1, 1)
    j_idx = cp.arange(levels).reshape(1, levels, 1, 1)
    gl1_expand = gl1.reshape(1, 1, h, w)
    gl2_expand = gl2.reshape(1, 1, h, w)

    glcm = ((gl1_expand == i_idx) & (gl2_expand == j_idx)).astype(cp.float32)
    glcm = ndimage.uniform_filter(glcm, size=(1, 1, ks, ks), mode="reflect") * (ks * ks)

    return glcm


def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=5, distance=1.0, angle=0.0):
    if HAS_CUPY:
        return fast_glcm_gpu(img, vmin, vmax, levels, kernel_size, distance, angle)
    else:
        return fast_glcm_cpu(img, vmin, vmax, levels, kernel_size, distance, angle)


def fast_glcm_contrast(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    is_gpu = HAS_CUPY and isinstance(glcm, cp.ndarray)
    xp = cp if is_gpu else np

    cont = xp.zeros((h, w), dtype=xp.float32)
    for i in range(levels):
        for j in range(levels):
            val = float((i - j) ** 2) if is_gpu else (i - j) ** 2
            cont += glcm[i, j] * val

    return cp.asnumpy(cont) if is_gpu else cont


def fast_glcm_homogeneity(
    img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0
):
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    is_gpu = HAS_CUPY and isinstance(glcm, cp.ndarray)
    xp = cp if is_gpu else np

    homo = xp.zeros((h, w), dtype=xp.float32)
    for i in range(levels):
        for j in range(levels):
            val = float(1.0 + (i - j) ** 2) if is_gpu else (1.0 + (i - j) ** 2)
            homo += glcm[i, j] / val

    return cp.asnumpy(homo) if is_gpu else homo


def fast_glcm_entropy(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    is_gpu = HAS_CUPY and isinstance(glcm, cp.ndarray)
    xp = cp if is_gpu else np

    pnorm = glcm / (xp.sum(glcm, axis=(0, 1)) + 1e-10) + 1.0 / ks**2
    ent = xp.sum(-pnorm * xp.log(pnorm + 1e-10), axis=(0, 1))

    return cp.asnumpy(ent) if is_gpu else ent


def fast_glcm_ASM(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    is_gpu = HAS_CUPY and isinstance(glcm, cp.ndarray)
    xp = cp if is_gpu else np

    asm = xp.zeros((h, w), dtype=xp.float32)
    for i in range(levels):
        for j in range(levels):
            asm += glcm[i, j] ** 2

    res_energy = xp.sqrt(asm)
    if is_gpu:
        return cp.asnumpy(asm), cp.asnumpy(res_energy)
    return asm, res_energy

def fast_glcm_correlation(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    is_gpu = HAS_CUPY and isinstance(glcm, cp.ndarray)
    xp = cp if is_gpu else np

    pnorm = glcm / (xp.sum(glcm, axis=(0, 1)) + 1e-10)
    
    mean_x = xp.zeros((h, w), dtype=xp.float32)
    mean_y = xp.zeros((h, w), dtype=xp.float32)
    for i in range(levels):
        for j in range(levels):
            mean_x += i * pnorm[i, j]
            mean_y += j * pnorm[i, j]

    var_x = xp.zeros((h, w), dtype=xp.float32)
    var_y = xp.zeros((h, w), dtype=xp.float32)
    for i in range(levels):
        for j in range(levels):
            var_x += ((i - mean_x) ** 2) * pnorm[i, j]
            var_y += ((j - mean_y) ** 2) * pnorm[i, j]
            
    std_x = xp.sqrt(var_x)
    std_y = xp.sqrt(var_y)

    corr = xp.zeros((h, w), dtype=xp.float32)
    for i in range(levels):
        for j in range(levels):
            corr += (i - mean_x) * (j - mean_y) * pnorm[i, j] / (std_x * std_y + 1e-10)

    return cp.asnumpy(corr) if is_gpu else corr

