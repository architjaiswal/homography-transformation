
import cv2
import numpy as np

def computeHomography(Us, Vs):
    """
    Estimate the homography matrix from given 4 points
    """
    # form the matrix AH = 0, then solve for H using SVD
    aList = []
    
    # your code
    for i in range(4):
        x = Us[i][0]
        y = Us[i][1]
        x_dash = Vs[i][0]
        y_dash = Vs[i][1]
        aList.append([x, y, 1, 0, 0, 0, -x*x_dash, -y*x_dash, -x_dash])
        aList.append([0, 0, 0, x, y, 1, -x*y_dash, -y*y_dash, -y_dash])

    matrixA = np.matrix(aList)

    # SVD  composition
    u, s, v = np.linalg.svd(matrixA)
    # reshape the rightmost singular vector into a 3x3
    H = np.reshape(v[8], (3, 3))

    #normalize and now we have H
    H = (1/H.item(8)) * H
    return H

# Original function from HH3\AR_Homography.py
# def warp_and_augment(im_logo, im_dst, H):
#     """
#     Given logo image, destination image, and the homography
#     Find the warped final output
#     """
#     imw, imh = im_dst.shape[1], im_dst.shape[0]
#     im_warped = cv2.warpPerspective(im_logo, H, (imw, imh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
#     # get mask for augmented image
#     mask = np.array(np.nonzero(im_warped))
#     im_out = im_dst.copy()
#     for n in range(mask.shape[1]):
#         i, j, k = mask[0, n], mask[1, n], mask[2, n]
#         im_out[i, j, k] = im_warped[i, j, k]
#     # There is a better way to do this: #todo for Bonus +5 point
#     return im_warped, im_out


# This is a better and modified implementation of warp_and_augment function
def warp_and_augment(im_logo, im_dst, H):
    """
    Given logo image, destination image, and the homography
    Find the warped final output
    """
    imw, imh = im_dst.shape[1], im_dst.shape[0]
    im_logo_warp = cv2.warpPerspective(im_logo, H, (imw, imh))

    # to remove noise, simply remove the pixels that sits in the place of logo in the destination image
    mask = cv2.threshold(cv2.cvtColor(im_logo_warp, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
    mask_inv = cv2.bitwise_not(mask)
    im_dst_bg = cv2.bitwise_and(im_dst, im_dst, mask=mask_inv)

    # After clearing the area of logo in the destination image, add the warped logo
    im_out = cv2.add(im_logo_warp, im_dst_bg)

    return im_logo_warp, im_out
