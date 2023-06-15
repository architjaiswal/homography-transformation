"""
EEL 4930/5934: Autonomous Robots
University Of Florida
"""
import cv2
import numpy as np

def click_event(event, x, y, flags, params):
    # call back function for mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        # put coordinates as text on the image
        cv2.putText(img, f'({x},{y})',(x,y),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
        # draw point on the image and print
        cv2.circle(img, (x,y), 3, (0,255, 255), -1)
        print(f'({x},{y})')


# read the input image
img = cv2.imread('data/game_fl.jpg')

# create a window
cv2.namedWindow('Point Coordinates')

# bind the callback function to window
cv2.setMouseCallback('Point Coordinates', click_event)

# display the image and get user input in loop
while True:
   cv2.imshow('Point Coordinates',img)
   k = cv2.waitKey(1) & 0xFF
   if k == 27: # untill ESC
      break
cv2.destroyAllWindows()