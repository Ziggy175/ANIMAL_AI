from glob import glob                                                           
import cv2 
import os
pngs = glob('raw-img/*/**.jpg')

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpeg', img)
    if not cv2.imread(j[:-3]) == 'jpeg':
        os.remove(j)