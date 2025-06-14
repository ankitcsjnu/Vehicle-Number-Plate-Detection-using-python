import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt

def detect_number_plate(image_path):
    plate_cascade = cv2.CascadeClassifier(r"C:\Users\ankit\OneDrive\Desktop\project design lab project\number plate\haarcascade_russian_plate_number.xml")
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load image. Please check the file path.")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in plates:
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        cropped_plate = gray[y:y+h, x:x+w]

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_plate)

        for detection in result:
            text = detection[1]
            print("Detected Number Plate:", text)

            cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # Hide axis
    plt.title("Number Plate Detection")
    plt.show()


image_path = r"C:\Users\ankit\OneDrive\Desktop\project design lab project\number plate\car img.jpg"
detect_number_plate(image_path)
