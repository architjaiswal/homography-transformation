"""
EEL 4930/5934: Autonomous Robots
University Of Florida
"""

import cv2
import numpy as np

from libs_hh3.homography import computeHomography, warp_and_augment

if __name__ == '__main__' : 
    # Read source image.
    im_logo = cv2.imread('data/logo_Gators.png') 

    # Corners of the logo: (0, 0), (w, h), (w, 0), (0, h)
    pts_logo = np.array([[0, 0],[500, 280],[500, 0],[0, 280]])

    # Read destination image.
    im_dst = cv2.imread('data/game_fl.jpg')

    # Destination points (use Get_mouse_clicks.py to get these points)
    #pts_dst = np.array([[937, 200], [1063, 252], [1061, 186], [937, 262]])
    #pts_dst = np.array([[761, 500], [990, 623], [1105, 513], [520, 580]])
    pts_dst = np.array([[325, 642], [1103, 512],[754, 501], [900, 724]])

    # Calculate Homography
    ##H, status = cv2.findHomography(pts_logo, pts_dst) # for check only
    H = computeHomography(pts_logo, pts_dst) # implement this function for HW

    # Warp source image to destination based on homography
    im_warped, im_out = warp_and_augment(im_logo, im_dst, H)

    # write output images
    cv2.imwrite("data/AR_warped_original.jpg", im_warped)
    cv2.imwrite("data/AR_out_original.jpg", im_out)
 

