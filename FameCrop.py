import cv2
import random
import os

i = 0
file_dir = './data/train/screenShots/'
if not os.path.exists('./data/train/Cartoon/'):
    os.makedirs('./data/train/Cartoon/')

for _, _,files in os.walk(file_dir):
    for i in range(len(files)):
        #read images
        path = file_dir + files[i]
        img=cv2.imread(path)
        sp = img.shape

        #h、w is the size of screenshot
        h = sp[0]
        w = sp[1]
        #randomly generate x
        x = random.randint(1, w - h)
        cropImg = img[0:h, (x):(x + h),:]
        if i%50 == 0:
            print(i)
            print('the crop position x:',x)
        img = cv2.resize(cropImg, (256, 256), interpolation = cv2.INTER_AREA)
        cv2.imwrite('./data/train/Cartoon/' + str(i).zfill(5) + '.jpg', img)
    