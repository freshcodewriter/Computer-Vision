import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from mymosaic import mymosaic

if __name__ == "__main__":

    # Loading franklin field picture
    # img_ref = mpimg.imread("./img/franklin-field/middle.jpg")
    # img_wrap_l = mpimg.imread("./img/franklin-field/left.jpg")
    # img_wrap_r = mpimg.imread("./img/franklin-field/right.jpg")
    # res_1 = mymosaic([img_wrap_l, img_ref, img_wrap_r])
    # plt.imshow(res_1)
    plt.savefig('./res/filed.png', dpi=300)
    # plt.show()

    # Load our own pictures taken from Towne building.
    img_ref = mpimg.imread("./img/towne/middle.JPG")
    img_wrap_l = mpimg.imread("./img/towne/left.JPG")
    img_wrap_r = mpimg.imread("./img/towne/right.JPG")
    res_2 = mymosaic([img_wrap_l, img_ref, img_wrap_r])
    plt.imshow(res_2)
    plt.savefig('./res/towne.png', dpi=300)
    plt.show()