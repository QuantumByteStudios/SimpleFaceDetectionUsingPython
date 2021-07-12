import cv2

introtext = ('''
 _____              ____       _            _
|  ___|_ _  ___ ___|  _ \  ___| |_ ___  ___| |_ ___  _ __
| |_ / _` |/ __/ _ \ | | |/ _ \ __/ _ \/ __| __/ _ \| '__|
|  _| (_| | (_|  __/ |_| |  __/ ||  __/ (__| || (_) | |
|_|  \__,_|\___\___|____/ \___|\__\___|\___|\__\___/|_|

Detects Faces With accuracy of 80% 
\n\nCreated By Harshit Raheja and Sumesh Mohanty\n\n_______________________________________________________________
	''')
print(introtext)

#Get input of Image name or Path of image
image_name_with_extension = input("Enter the Name or Path of Image: ")



# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread(image_name_with_extension)

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#Important Variables For Font
fontcolor = (255, 51, 0) # DarkBlue color in BGR
fontthickness = 1 # Line thickness of 1 px
fontScale = 0.6 # fontScale
fontcord = (30, 30) # font co-ordinates X = 50, y = 50
font = cv2.FONT_HERSHEY_SIMPLEX # font

# Using cv2.putText() method
image = cv2.putText(img, 'FaceDetected!', fontcord, font, fontScale, fontcolor, fontthickness, cv2.LINE_AA)



# Prints Image Details
print("\n")
print(f"Height, Width and Number of Channels in image(RGB)  is: {img.shape}")



# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (89, 179, 3), 3)






# Display the output
cv2.imshow('img', img)
cv2.waitKey(30000)
