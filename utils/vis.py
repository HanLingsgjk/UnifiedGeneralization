import numpy as np
from matplotlib import cm
import cv2

def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    if len(x.shape) != len(w.shape):
        w = np.broadcast_to(w[..., None], x.shape)
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)


def sinebow(h):
    """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
    f = lambda x: np.sin(np.pi * x) ** 2
    return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def matte(vis, acc, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :])
    bg = np.where(bg_mask, light, dark)
    return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]


def visualize_cmap(value,
                   weight,
                   colormap,
                   lo=None,
                   hi=None,
                   percentile=99.,
                   curve_fn=lambda x: x,
                   modulus=None,
                   matte_background=True):
    """Visualize a 1D image and a 1D weighting according to some colormap.

  Args:
    value: A 1D image.
    weight: A weight map, in [0, 1].
    colormap: A colormap function.
    lo: The lower bound to use when rendering, if None then use a percentile.
    hi: The upper bound to use when rendering, if None then use a percentile.
    percentile: What percentile of the value map to crop to when automatically
      generating `lo` and `hi`. Depends on `weight` as well as `value'.
    curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
    modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
      `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
    matte_background: If True, matte the image over a checkerboard.

  Returns:
    A colormap rendering.
  """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(
        value, weight, [50 - percentile / 2, 50 + percentile / 2], assume_sorted=True)

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = np.finfo(np.float32).eps
    lo = lo or (lo_auto - eps)
    hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
            np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

    if colormap:
        colorized = colormap(value)[:, :, :3]
    else:
        if len(value.shape) != 3:
            raise ValueError(f'value must have 3 dims but has {len(value.shape)}')
        if value.shape[-1] != 3:
            raise ValueError(
                f'value must have 3 channels but has {len(value.shape[-1])}')
        colorized = value

    return matte(colorized, weight) if matte_background else colorized


def visualize_coord_mod(coords, acc):
    """Visualize the coordinate of each point within its "cell"."""
    return matte(((coords + 1) % 2) / 2, acc)


def make_color_wheel(bins=None):
    """Build a color wheel.
    Args:
        bins(list or tuple, optional): Specify the number of bins for each
            color range, corresponding to six ranges: red -> yellow,
            yellow -> green, green -> cyan, cyan -> blue, blue -> magenta,
            magenta -> red. [15, 6, 4, 11, 13, 6] is used for default
            (see Middlebury).
    Returns:
        ndarray: Color wheel of shape (total_bins, 3).
    """
    if bins is None:
        bins = [15, 6, 4, 11, 13, 6]
    assert len(bins) == 6

    RY, YG, GC, CB, BM, MR = tuple(bins)
    print(RY)
    ry = [1, np.arange(RY) / RY, 0]
    yg = [1 - np.arange(YG) / YG, 1, 0]
    gc = [0, 1, np.arange(GC) / GC]
    cb = [0, 1 - np.arange(CB) / CB, 1]
    bm = [np.arange(BM) / BM, 0, 1]
    mr = [1, 0, 1 - np.arange(MR) / MR]
    print(ry)
    num_bins = RY + YG + GC + CB + BM + MR
    print(num_bins)
    color_wheel = np.zeros((3, num_bins), dtype=np.float32)
    print(color_wheel)

    col = 0
    for i, color in enumerate([ry, yg, gc, cb, bm, mr]):
        if i == 0:
            print(i, color)
        for j in range(3):
            color_wheel[j, col:col + bins[i]] = color[j]
        col += bins[i]

    return color_wheel.T

def flow2rgb(flow, color_wheel=None, unknown_thr=1e6):
    """Convert flow map to RGB image.

    Args:
        flow (ndarray): Array of optical flow.
        color_wheel (ndarray or None): Color wheel used to map flow field to
            RGB colorspace. Default color wheel will be used if not specified.
        unknown_thr (str): Values above this threshold will be marked as
            unknown and thus ignored.

    Returns:
        ndarray: RGB image that can be visualized.
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]
    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (
        np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |
        (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx**2 + dy**2)  # HxW
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)  # 使用最大模长来放缩坐标值
        dx /= max_rad
        dy /= max_rad

    rad = np.sqrt(dx**2 + dy**2)  # HxW
    angle = np.arctan2(-dy, -dx) / np.pi  # HxW（-1, 1]

    bin_real = (angle + 1) / 2 * (num_bins - 1)  # HxW (0, num_bins-1]
    bin_left = np.floor(bin_real).astype(int)  # HxW 0,1,...,num_bins-1
    bin_right = (bin_left + 1) % num_bins  # HxW 1,2,...,num_bins % num_bins -> 1, 2, ..., num_bins, 0
    w = (bin_real - bin_left.astype(np.float32))[..., None]  # HxWx1
    flow_img = (1 - w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]  # 线性插值计算实际的颜色值
    small_ind = rad <= 1  # 以模长为1作为分界线来分开处理，个人理解这里主要是用来控制颜色的饱和度，而前面的处理更像是控制色调。
    # 小于1的部分拉大
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    # 大于1的部分缩小
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0

    return flow_img
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
def grayscale_cmap(cmap):
    """返回给定色图的灰度版本"""
    cmap = plt.cm.get_cmap(cmap) # 使用名称获取色图对象
    colors = cmap(np.arange(cmap.N)) # 将色图对象转为RGBA矩阵，形状为N×4

    # 将RGBA颜色转换为灰度
    # 参考 http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114] # RGB三色的权重值
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight)) # RGB平方值和权重的点积开平方根
    colors[:, :3] = luminance[:, np.newaxis] # 得到灰度值矩阵
    # 返回相应的灰度值色图
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)


def view_colormap(cmap):
    """将色图对应的灰度版本绘制出来"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots(2, figsize=(6, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])