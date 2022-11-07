import pathlib

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def read_points(filename):
    points = []
    with open(filename, encoding='utf-8') as f:
        next(f)
        for line in f:
            points.append([float(c) for c in line.split()] + [1.])
    return np.array(points)


def eight_point(pt1, pt2):
    uv_matrices = (pt1[:, :, np.newaxis] @ pt2[:, np.newaxis]).reshape((-1, 9))

    f_hat = svd_least_square(uv_matrices)
    fundamental_mat = enforce_rank2(f_hat)
    return fundamental_mat


def svd_least_square(uv_matrices):
    _, _, vt = np.linalg.svd(uv_matrices)
    f = vt[-1].reshape((3, 3))
    return f


def enforce_rank2(f_hat):
    u, d, v = np.linalg.svd(f_hat)
    d[-1] = 0
    f = u @ np.diag(d) @ v
    return f


def epipolar_lines_from_f(f, pt, pt2):
    lines = f @ pt[..., np.newaxis]
    a, b, c = np.split(lines[..., 0], 3, axis=1)

    start_pts = np.hstack((np.zeros_like(a), -c / b))
    end_pts = np.hstack((np.ones_like(a) * 512, -(c + a * 512) / b))
    epi_lines = np.stack((start_pts, end_pts), axis=1).astype(int)

    distances = np.abs(a * pt2[:, 0, np.newaxis] + b * pt2[:, 1, np.newaxis] + c) / np.hypot(a, b)

    return epi_lines, distances.mean()


def draw_epilines(fname, pts, epipolar_lines):
    image = cv2.imread(fname)
    cv2.polylines(image, epipolar_lines, False, (255, 0, 0))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.scatter(pts[..., 0], pts[..., 1], s=50)
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[..., ::-1]

    return data


if __name__ == '__main__':
    matplotlib.use('agg')

    pts1 = read_points('pt_2D_1.txt')
    pts2 = read_points('pt_2D_2.txt')

    output_dir = pathlib.Path('output')
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print('(a) eight-point algorithm')
    F = eight_point(pts1, pts2)
    print(f'Fundamental matrix = \n{F}\n')

    epi_lines1, mean_dist1 = epipolar_lines_from_f(F, pts2, pts1)
    print(f'mean distance between points and epipolar lines in image 1: {mean_dist1:.4f}')
    img = draw_epilines('image1.jpg', pts1, epi_lines1)
    cv2.imwrite(str(output_dir / 'a_img1.jpg'), img)

    epi_lines2, mean_dist2 = epipolar_lines_from_f(F.T, pts1, pts2)
    print(f'mean distance between points and epipolar lines in image 2: {mean_dist2:.4f}\n')
    img = draw_epilines('image2.jpg', pts2, epi_lines2)
    cv2.imwrite(str(output_dir / 'a_img2.jpg'), img)
