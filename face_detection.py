import cv2
import os


# Take user input for the image path
imgPath = input('Enter the image path: ')
# A directory path to save the headshot(s)
headshotPath = input('Enter the directory path to save the headshot(s): ')
# Create the directory if it doesn't exist
if not os.path.exists(headshotPath):
    os.makedirs(headshotPath)
# Load the cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread(imgPath)
# Convert into grayscale
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = faceCascade.detectMultiScale(grayImg, scaleFactor=1.05, minNeighbors=5)
# Draw rectangle around the faces
for i, (x, y, w, h) in enumerate(faces):
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # Save the headshot
    cv2.imwrite(f'{headshotPath}/face_{i + 1}.jpg', img[y:y + h, x:x + w])
# Display the output
print(f'The number of faces in the image is {len(faces)}')

# I didn't need numpy and PIL
