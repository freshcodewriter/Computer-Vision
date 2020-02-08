import numpy as np
import cv2
import dlib
import scipy
from scipy.spatial import Delaunay


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def get_convexhull(predictor, img_gray, face):
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = detector(img_gray)
    # for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)

    return points, landmarks_points, convexhull

def de_triangulation(detector, predictor, img, mask):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray)
    for face in faces:
        points, landmarks_points, convexhull = get_convexhull(predictor, img_gray, face)
        # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
        cv2.fillConvexPoly(mask, convexhull, 255)

        face_image_1 = cv2.bitwise_and(img, img, mask=mask)

        # Delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)

    return indexes_triangles, landmarks_points


# def triangulation_both_faces(landmarks_points, landmarks_points2, triangle_index, img):
#     tr1_pt1 = landmarks_points[triangle_index[0]]
#     tr1_pt2 = landmarks_points[triangle_index[1]]
#     tr1_pt3 = landmarks_points[triangle_index[2]]
#     triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

#     rect1 = cv2.boundingRect(triangle1)
#     (x, y, w, h) = rect1
#     cropped_triangle = img[y: y + h, x: x + w]
#     cropped_tr_mask = np.zeros((h, w), np.uint8)


#     points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
#                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
#                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

#     cv2.fillConvexPoly(cropped_tr_mask, points, 255)

#     tr2_pt1 = landmarks_points2[triangle_index[0]]
#     tr2_pt2 = landmarks_points2[triangle_index[1]]
#     tr2_pt3 = landmarks_points2[triangle_index[2]]
#     triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


#     rect2 = cv2.boundingRect(triangle2)
#     (x, y, w, h) = rect2

#     cropped_tr2_mask = np.zeros((h, w), np.uint8)

#     points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
#                         [tr2_pt2[0] - x, tr2_pt2[1] - y],
#                         [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

#     cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

#     # Warp triangles
#     points = np.float32(points)
#     points2 = np.float32(points2)
#     M = cv2.getAffineTransform(points, points2)
#     warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
#     warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

#     return warped_triangle, x, y, w, h

def triangulation_both_faces(face1, face2, tr1, tr2):
    rect1 = cv2.boundingRect(tr1)
    rect2 = cv2.boundingRect(tr2)
    tr1_pt = []
    tr2_pt = []
    for triangle_index in range(3):
        tr1_pt.append(((tr1[triangle_index][0] - rect1[0]),(tr1[triangle_index][1] - rect1[1])))
        tr2_pt.append(((tr2[triangle_index][0] - rect2[0]),(tr2[triangle_index][1] - rect2[1])))
    cropped_triangle = face1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    points = np.float32(tr1_pt)
    points2 = np.float32(tr2_pt)
    M = cv2.getAffineTransform(points, points2)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (rect2[2], rect2[3]),None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((rect2[3], rect2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tr2_pt), (1.0, 1.0, 1.0), 16, 0);
    warped_triangle = warped_triangle * mask
    # Copy triangular region of the rectangular patch to the output image
    face2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = face2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] * ( (1.0, 1.0, 1.0) - mask )
    face2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = face2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] + warped_triangle

    
def Blend_faces(new_weighted, shape_cap2, image, image2):
    beforeshape = new_weighted
    shape1 = np.append(new_weighted,np.array([[0,0],[0,image.shape[0]-1],[image.shape[1]-1,0], [image.shape[1]-1,image.shape[0]-1]]), axis=0)
    shape_cap2 = np.append(shape_cap2,np.array([[0,0],[0,image2.shape[0]-1],[image2.shape[1]-1,0], [image2.shape[1]-1,image2.shape[0]-1]]), axis=0)

    Tri1 = Delaunay(shape1)
    Tri2 = Delaunay(shape_cap2)
    warped_image=np.copy(image)
    for tri in Tri1.simplices:
        triangulation_both_faces(image2,warped_image,shape_cap2[tri],shape1[tri])
        
    hullIndex = cv2.convexHull(beforeshape, returnPoints = False)
    hull2 =[]
    i=0
    for i in range(0, len(hullIndex)):
        hull2.append(beforeshape[hullIndex[i]][0])
    hullmask = []
    i=0
    hull2 = np.asarray(hull2)
    for i in range(len(hull2)):
        hullmask.append((hull2[i,0], hull2[i,1]))
    
    mask = np.ones(image.shape, dtype = image.dtype)  
    cv2.fillConvexPoly(mask, np.int32(hullmask), (255, 255, 255))
    rect = cv2.boundingRect(np.float32([hull2]))    
    center = ((rect[0]+int(rect[2]/2), rect[1]+int(rect[3]/2)))
    blend_image = cv2.seamlessClone(warped_image, image, mask, center, cv2.NORMAL_CLONE)   
    
    return blend_image