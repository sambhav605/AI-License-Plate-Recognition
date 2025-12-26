import cv2 
import numpy as np # Standard practice to use 'np'
from ultralytics import YOLO
import math
import easyocr

## Video Input Setup 
cap = cv2.VideoCapture("./Public/test.mp4")

## Using Our Pretrained Model
model = YOLO('best.pt')
classNames = ['license-plate']

## Initializing EasyOCR 
# 'en' for English; gpu=True uses your graphics card if available
reader = easyocr.Reader(['en'], gpu=True)

## Function to get the license plate Number 
def preprocess_plate(plate_img):
    if plate_img.size == 0: 
        return None
    
    ## Converting to Gray
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    ## Apply CLAHE to fix uneven lighting 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)

    ## Binarization
    thresh = cv2.adaptiveThreshold(
        contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    ## Resizing for better Result 
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return resized

while True:
    success, img = cap.read()

    if not success or img is None:
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            ## Crop the plate
            plate_crop = img[y1:y2, x1:x2]

            ## Preprocessing 
            processed_plate = preprocess_plate(plate_crop)
            
            if processed_plate is not None:
                # EasyOCR returns: (bounding box, text, confidence)
                ocr_results = reader.readtext(processed_plate)

                for (bbox, text, conf) in ocr_results:
                    ## Filtering the text
                    clean_text = "".join(c for c in text if c.isalnum()).upper()

                    if conf > 0.4 and len(clean_text) > 4:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            img, 
                            f"{clean_text} ({math.ceil(conf*100)/100})", 
                            (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )

    cv2.imshow("Output", img)
    
    # Changed waitKey(0) to waitKey(1) so the video plays automatically
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()