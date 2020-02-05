import cv2
import random
import os

i = 0
file_dir = './data/train/screenShots/'
for _, _,files in os.walk(file_dir):
    #read images
    path = file_dir + files[i]
    i = i+1
    img=cv2.imread(path)
    sp = img.shape

    #h、w is the size of screenshot
    h = sp[0]
    w = sp[1]
    #randomly generate x
    x = random.randint(1, w - h)
    cropImg = img[0:h, (x):(x + h),:]
    if count%50 == 0:
        print(count)
        print('x:',x)
    img = cv2.resize(cropImg, (256, 256), interpolation = cv2.INTER_AREA)
    cv2.imwrite('./data/train/Cartoon/' + str(count + 2491).zfill(5) + '.jpg', img)
    