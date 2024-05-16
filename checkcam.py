import cv2

def list_available_cameras():
    # Iterate over camera indices starting from 0
    index = 0
    while True:
        # Try to open the camera
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        # Check if the camera is opened successfully
        if not cap.isOpened():
            break
        
        # Release the camera
        cap.release()
        
        # Print information about the camera
        print(f"Camera {index}: Opened successfully")
        
        # Move to the next camera index
        index += 1

# Call the function to list available cameras
list_available_cameras()
