import cv2
import glob, os

count = 0

for imgFile in glob.iglob('D:\\ASD Data\\Images\\Positive\\Original\\*.png', recursive=False):
    print(imgFile)
    img = cv2.imread(imgFile)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Thresholding
    th, threshed = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY)
    #Find the max-area contour
    _, contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea)[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    dst = img[y:y + h, x:x + w]#Cropped image

    imgID = os.path.basename(imgFile).split(".png")[0]
    cv2.imwrite("D:\\ASD Data\\Cropped\\Positive\\Original\\"+imgID+".jpg", dst)

    print("Image no.:", count)
    count = count + 1

print("Count of imgs:", count)