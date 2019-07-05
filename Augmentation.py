from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import glob

augmenter = ImageDataGenerator(
    rotation_range=10,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

imgCount = 10 #Count of synthetic images to be generagted for every image
print("Augmenting Negative Samples:")
for file in glob.iglob('D:\\Images\\Negative\\Original\\*.png'):
    print("Current File:", file)
    img = load_img(file)  # this is a PIL image
    img = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    img = img.reshape((1,) + img.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    participantID = file.split("_")[1].replace(".png", "")
    imgPrefix = "TCAug_" + participantID

    i = 0
    for batch in augmenter.flow(img, batch_size=1,
                              save_to_dir='D:\\Images\\Negative\\Augmented', save_prefix=imgPrefix, save_format='png'):
        i += 1
        if i == imgCount:
            break

#Augmenting TS images
print("Augmenting Positive Samples:")
for file in glob.iglob('D:\\Images\\Positive\\Original\\*.png'):
    print("Current File:", file)
    img = load_img(file)  # this is a PIL image
    img = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    img = img.reshape((1,) + img.shape)  #Numpy array with shape (1, 3, 150, 150)

    participantID = file.split("_")[1].replace(".png", "")
    imgPrefix = "TSAug_" + participantID

    i = 0
    for batch in augmenter.flow(img, batch_size=1,
                              save_to_dir='D:\\Images\\Positive\\Augmented', save_prefix=imgPrefix, save_format='png'):
        i += 1
        if i == imgCount:
            break