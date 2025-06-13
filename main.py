import cv2
from contrast import apply_contrast

def run_stream(source=0):
    video_capture = cv2.VideoCapture(source)

    if not video_capture.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Press 'q' to quit, 'c' to cycle through methods")

    contrast_methods = [
        "normal",
        "clahe",
        "histogram",
        "linear",
        "gamma",
        "binary"
    ]
    current_method = 0

    while True:
        key = cv2.waitKey(1)
        ret, frame = video_capture.read()

        if not ret:
            print("Frame not read successfully!")
            break

        frame = apply_contrast(frame, contrast_methods[current_method])

        #Add label
        cv2.putText(frame, contrast_methods[current_method].upper(), 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Contrast Changer', frame)

        if key == ord('q'):
            break
        elif key == ord('c'):
            current_method = (current_method + 1) % len(contrast_methods)

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_stream()