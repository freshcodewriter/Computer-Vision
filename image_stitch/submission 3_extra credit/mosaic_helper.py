import numpy as np
from scipy.ndimage import geometric_transform, map_coordinates
import matplotlib.pyplot as plt


def draft(img, H):
    h, w, num_channel = img.shape

    def map_left_on_canvas(H):
        # Map corners to homography space
        affine_corners = np.asarray([[0, 0, 1],
                                     [0, h - 1, 1],
                                     [w - 1, h - 1, 1],
                                     [w - 1, 0, 1]], dtype='int').transpose(1, 0)
        affine_target = np.matmul(H, affine_corners)
        affine_target = np.divide(affine_target, affine_target[2, :])
        # Set the boundary in homography space
        x_min, y_min = np.amin(affine_target[0:2, :], axis=1)
        x_max, y_max = np.amax(affine_target[0:2, :], axis=1)
        affine_boundary = np.asarray([[x_min, x_min, x_max, x_max],
                                      [y_min, y_max, y_max, y_min]])
        return affine_target[0:2, :].transpose(1, 0), affine_boundary.transpose(1, 0).astype('int')

    corners_canvas, boundary_canvas = map_left_on_canvas(H)

    def plot_draft():
        plt.imshow(img)
        plot_x = np.append(corners_canvas[:, 0], corners_canvas[0, 0])
        plot_y = np.append(corners_canvas[:, 1], corners_canvas[0, 1])
        boundary_x = np.append(boundary_canvas[:, 0], boundary_canvas[0, 0])
        boundary_y = np.append(boundary_canvas[:, 1], boundary_canvas[0, 1])
        plt.plot(plot_x, plot_y, marker="o")
        plt.plot(boundary_x, boundary_y, marker="o")
        plt.show()
    plot_draft()
    return boundary_canvas


def stitch(img_wrap, img_ref, b, w, h):
    print('----- stitch -----')
    x_min = b[0, 0]
    y_min = b[0, 1]
    x_max = b[2, 0]
    y_max = b[2, 1]
    num_channel = 3
    b_shape = (y_max - y_min + 1, x_max - x_min + 1)
    # print(b_shape)
    # print(img_wrap.shape)
    # print('x_min: ', x_min)
    # print('y_min: ', y_min)
    # print('x_max: ', x_max)
    # print('y_max: ', y_max)

    canvas_h = max(0, -y_min) + max(h, y_max) + 1

    if x_min < 0:
        canvas_w = w - x_min + 1
        if y_min < 0:
            wrap_ori = [0, 0]
            ref_ori = [-x_min, -y_min]
        else:
            wrap_ori = [0, y_min]
            ref_ori = [-x_min, 0]
    else:
        canvas_w = x_max + 1
        if y_min < 0:
            wrap_ori = [x_min, 0]
            ref_ori = [0, -y_min]
        else:
            # wrap_ori = [canvas_w - img_wrap.shape[1], canvas_h - img_wrap.shape[0]]
            wrap_ori = [x_min, y_min]
            ref_ori = [0, 0]

    # print(ref_ori)
    # print(wrap_ori)
    canvas = np.zeros((canvas_h, canvas_w, num_channel), dtype="uint8")
    canvas_tmp = canvas.copy()
    canvas[ref_ori[1]:ref_ori[1] + img_ref.shape[0], ref_ori[0]:ref_ori[0] + img_ref.shape[1], :] = img_ref
    canvas_tmp[wrap_ori[1]:wrap_ori[1] + img_wrap.shape[0], wrap_ori[0]:img_wrap.shape[1] + wrap_ori[0], :] = img_wrap
    canvas[canvas == 0] = canvas_tmp[canvas == 0]
    return canvas, ref_ori, wrap_ori


def get_imgs(img_wrap, img_ref, b):
    x_min = b[0, 0]
    y_min = b[0, 1]
    x_max = b[2, 0]
    y_max = b[2, 1]
    h, w, num_channel = img_ref.shape
    b_shape = (y_max - y_min + 1, x_max - x_min + 1)
    canvas_h = max(0, -y_min) + max(h, y_max) + 1

    if x_min < 0:
        canvas_w = w - x_min + 1
        if y_min < 0:
            wrap_ori = [0, 0]
            ref_ori = [-x_min, -y_min]
        else:
            wrap_ori = [0, y_min]
            ref_ori = [-x_min, 0]
    else:
        canvas_w = x_max + 1
        if y_min < 0:
            wrap_ori = [x_min, 0]
            ref_ori = [0, -y_min]
        else:
            wrap_ori = [x_min, y_min]
            ref_ori = [0, 0]

    canvas_ref = np.zeros((canvas_h, canvas_w, num_channel), dtype="uint8")
    canvas_wrap = canvas_ref.copy()
    canvas_ref[ref_ori[1]:ref_ori[1] + img_ref.shape[0], ref_ori[0]:ref_ori[0] + img_ref.shape[1], :] = img_ref
    canvas_wrap[wrap_ori[1]:wrap_ori[1] + img_wrap.shape[0], wrap_ori[0]:img_wrap.shape[1] + wrap_ori[0], :] = img_wrap
    res_img = [canvas_ref, canvas_wrap]

    return res_img


def wrap_img(img, b, H):
    x_min = b[0, 0]
    y_min = b[0, 1]
    x_max = b[2, 0]
    y_max = b[2, 1]
    x_range = np.arange(x_min, x_max + 1)
    y_range = np.arange(y_min, y_max + 1)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    b_shape = (y_max - y_min + 1, x_max - x_min + 1)

    # Affine transformation
    affine_x = x_mesh.flatten()
    affine_y = y_mesh.flatten()
    affine_src = np.stack((affine_x, affine_y, np.ones_like(affine_y)))
    affine_target = np.matmul(np.linalg.inv(H), affine_src)
    affine_target = np.divide(affine_target, affine_target[2, :])
    coordinates = affine_target[0:2, :]
    # Calculate and stack each channel
    img_trans_r = map_coordinates(img[:, :, 0].transpose(1, 0), coordinates).reshape(b_shape)
    img_trans_g = map_coordinates(img[:, :, 1].transpose(1, 0), coordinates).reshape(b_shape)
    img_trans_b = map_coordinates(img[:, :, 2].transpose(1, 0), coordinates).reshape(b_shape)
    affine_img = np.dstack((img_trans_r, img_trans_g, img_trans_b))
    return affine_img
