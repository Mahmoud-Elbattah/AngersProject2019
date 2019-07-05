import numpy as np
from scipy import misc
import glob

negLabel = 0.0
posLabel = 1.0
imgDimension = 150

f_handle = open('D:\\ASD_Cropped.csv', 'a')

headerRow = "Label"
for i in range (0,(imgDimension*imgDimension)):
    headerRow = headerRow + ",Pixel" + str(i)

headerRow = headerRow + "\n"
f_handle.write(headerRow)
print("Header Done.")

for file in glob.iglob('D:\\ASD Data\\Cropped\\Negative\\**\\*.jpg', recursive=True):
    print("Current File:", file)

    img = misc.imread(file)
    img = img[: ,:, :3]#Excluding the Alpha channel
    img = misc.imresize(img, size=(imgDimension, imgDimension))

    img = (img / 255.0)#Normalize
    img = img.reshape(-1, 3)

    #Converting into grayscale
    red = img[:,0]
    green = img[:,1]
    blue = img[:,2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue

    #Adding the label
    gray = np.insert(gray, 0, negLabel, axis=0)

    gray = gray.reshape(1, -1)
    np.savetxt(f_handle, gray, fmt='%.4g', delimiter=",")


print("Negative Done.")


for file in glob.iglob('D:\\ASD Data\\Cropped\\Positive\\**\\*.jpg', recursive=True):
    print("Current File:", file)

    img = misc.imread(file)
    img = img[:,:, :3]#Excluding the Alpha channel
    img = misc.imresize(img, size=(imgDimension, imgDimension))

    img = (img / 255.0)#Normalize
    img = img.reshape(-1, 3)

    #Converting into grayscale
    red = img[:,0]
    green = img[:,1]
    blue = img[:,2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue

    #Adding the participant info and label
    gray = np.insert(gray, 0, posLabel, axis=0)

    gray = gray.reshape(1, -1)
    np.savetxt(f_handle, gray, fmt='%.4g', delimiter=",")

print("Positive Done.")