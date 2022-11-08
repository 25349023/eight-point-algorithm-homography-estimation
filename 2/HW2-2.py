import itertools
import pathlib

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def svd_least_square(a_matrix):
    _, _, vt = np.linalg.svd(a_matrix)
    x = vt[-1].reshape((3, 3))
    return x


def relation_matrix(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return np.array(
        [[x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2],
         [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]]
    )


def warp_grid(homography, target_shape):
    width, height = target_shape[:2]
    pixel_indices = np.array(list(itertools.product(range(width), range(height), [1])))

    warp_matrix = (homography @ pixel_indices[..., np.newaxis]).reshape((-1, 3))
    warp_matrix = (warp_matrix / warp_matrix[:, 2:])

    warp_pixel = warp_matrix.astype(int)
    warp_diff = warp_matrix % 1
    return warp_matrix, warp_pixel, warp_diff


# def bilinear_pixel_color(src, x, y):
#     if x < 0 or x >= src.shape[0] - 1 or y < 0 or y >= src.shape[1] - 1:
#         return np.array([0, 0, 0])
#
#     px, py = int(x), int(y)
#     a = src[px, py]
#     b = src[px, py + 1]
#     c = src[px + 1, py]
#     d = src[px + 1, py + 1]
#     dx, dy = x % 1, y % 1
#
#     return (a * dy + b * (1 - dy)) * dx + (c * dy + d * (1 - dy)) * (1 - dx)


# [TODO] speed up bilinear interpolation

def bilinear_pixel_color(src, px, py, dx, dy):
    if px < 0 or px >= src.shape[0] - 1 or py < 0 or py >= src.shape[1] - 1:
        return np.array([0, 0, 0])

    # px, py = int(x), int(y)
    a = src[px, py]
    b = src[px, py + 1]
    c = src[px + 1, py]
    d = src[px + 1, py + 1]
    # dx, dy = x % 1, y % 1

    return (a * dy + b * (1 - dy)) * dx + (c * dy + d * (1 - dy)) * (1 - dx)


def draw_selected_polygon(image, polygon):
    image = image.copy()
    polygon = np.array(polygon, dtype=int)[..., ::-1]

    cv2.polylines(image, [polygon], True, (0, 0, 255), 5)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(image)
    ax.scatter(polygon[..., 0], polygon[..., 1], s=50, c='y')
    fig.canvas.draw()

    return get_figure_rgb_data(fig)


def get_figure_rgb_data(fig):
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


if __name__ == '__main__':
    matplotlib.use('agg')

    output_dir = pathlib.Path('output')
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    pts1 = [[100, 200], [100, 1400], [1000, 1400], [1000, 200]]
    pts2 = [[311, 450], [12, 891], [1000, 895], [812, 436]]

    A = np.vstack([relation_matrix(p, q) for p, q in zip(pts1, pts2)])

    H = svd_least_square(A)
    # H is target -> source
    print(f'homography = \n{H}')

    source = cv2.imread('Delta-Building.jpg')
    selected_img = draw_selected_polygon(source, pts2)
    cv2.imwrite(str(output_dir / 'selected_img.jpg'), selected_img)

    target = np.zeros_like(source)
    W, Wp, dW = warp_grid(H, target.shape)

    for m in tqdm.tqdm(range(W.shape[0]), desc='warping...'):
        warp_pt = Wp[m]
        warp_dpt = dW[m]
        i, j = divmod(m, target.shape[1])
        target[i, j] = bilinear_pixel_color(source, warp_pt[0], warp_pt[1],
                                            warp_dpt[0], warp_dpt[1])

    cv2.imwrite(str(output_dir / 'rectified_img.jpg'), target)
