import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from corner_detector import corner_detector
from anms import anms
from testing_anms import testing_anms
from feat_desc import feat_desc
from feat_match import feat_match
from crossMatch import crossMatch
from our_feat_desc import our_feat_desc
from matplotlib.patches import ConnectionPatch
from utils import rgb2gray


def feature_matching(img_ref, img_wrap, suffix):
    img_ref_gray = rgb2gray(img_ref)
    img_wrap_gray = rgb2gray(img_wrap)

    # Step 1 && Step 2
    print('[1] Corner Detecting...')
    cimg_1 = corner_detector(img_ref_gray)
    cimg_2 = corner_detector(img_wrap_gray)
    x_1, y_1, r_max_1 = testing_anms(cimg_1, 2000)
    x_2, y_2, r_max_2 = testing_anms(cimg_2, 2000)
    x_1 = np.asarray(x_1)
    y_1 = np.asarray(y_1)
    x_2 = np.asarray(x_2)
    y_2 = np.asarray(y_2)

    fig, (ax, ay) = plt.subplots(1, 2, sharey=True)
    ax.imshow(img_ref)
    ay.imshow(img_wrap)
    ax.plot(x_1, y_1, color='r', marker='o',
            linestyle='None', markersize=1)
    ay.plot(x_2, y_2, color='r', marker='o',
            linestyle='None', markersize=1)
    plt.show()

    # Step 3 & 4
    print('[2] Describing feature...')
    descs1 = feat_desc(img_ref_gray, x_1, y_1)
    descs2 = feat_desc(img_wrap_gray, x_2, y_2)

    print('[3] Matching feature...')
    match = feat_match(descs1, descs2)
    idx_1 = np.argwhere(match > -1)
    idx_1 = idx_1.flatten()
    idx_2 = np.take(match, idx_1)
    idx_2 = idx_2.astype(int)

    draw_x1 = x_1[idx_1]
    draw_y1 = y_1[idx_1]
    draw_x2 = x_2[idx_2]
    draw_y2 = y_2[idx_2]

    # Display Result
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img_ref)
    ax1.plot(draw_x1, draw_y1, color='r', marker='o',
             linestyle='None', markersize=1)

    ax2.imshow(img_wrap)
    ax2.plot(draw_x2, draw_y2, color='r', marker='o',
             linestyle='None', markersize=1)

    for i in range(draw_x1.size):
        xy1 = (draw_x1[i], draw_y1[i])
        xy2 = (draw_x2[i], draw_y2[i])
        con = ConnectionPatch(xyA=xy2, xyB=xy1, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color='#53F242')
        ax2.add_artist(con)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return draw_x1, draw_y1, draw_x2, draw_y2

