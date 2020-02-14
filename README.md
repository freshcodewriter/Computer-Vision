# Computer-Vision
Set of Computer Vision Projects on Classic Computer Vision Algorithm &amp; Modern CNN 
This repository consist of different projects I done for my computer vision class at Penn.

1. Edge Detection: 
- Apply Gaussian filter to smooth the image in order to remove the noise
- Find the intensity gradients of the image
- Apply non-maximum suppression to get rid of spurious response to edge detection
- Apply double threshold to determine potential edges
- Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.

2. Face Morphing:
A combination of generalized image warping with a cross-dissolve between pixels
- Select corresponding points in two images
- Pre-warp the two images
- Cross-dissolve their colors

3. Image Blending:
- Read in the source, mask, and target images.
- Expand the boundaries of the source and mask, based on the given offset values, to ensure that each input image is the same size.
- Produce the sparse matrix A by pre-computing the indeces and values for each of its elements, to emulate the unknown discrete laplacian gradients over the masked region of the final image (x).
- Produce the vector b by evaluating the discrete laplacian over the source image, bounded by the mask region, and combining those values with known pixel values of the target image.
- Knowing the formulations of A and b, solve for x.
- Clip pixel values of x that extend outside the valid intensity range and reshape to the proper image dimensions.

4. Image Stitch
- Detecting keypoints (DoG, Harris, etc.) and extracting local invariant descriptors (SIFT, SURF, etc.) from two input images
- Matching the descriptors between the images
- Using the RANSAC algorithm to estimate a homography matrix using our matched feature vectors
- Applying a warping transformation using the homography matrix obtained from Step #3

5. Seam Curving
- Calculate energy map
- Find minimum seam from top to bottom edge
- Remove minimum seam from top to bottom edge
- Repeat Steps 1 - 3 until desired number of seams are removed
- Repeat Steps 1 - 4 for left to right edge

6. Optical Flow
- Setting up your environment
- Shi-Tomasi Corner Detector - selecting the pixels to track
- Tracking Specific Objects
- Visualizing

7. Face Swapping in Video
A project that integrates all algorithm and swaps face in two videos

8. Deep Learning
Explore deep learning algorithm in CNN 
