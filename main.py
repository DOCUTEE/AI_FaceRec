import cv2
import face_recognition
import numpy as np
import os

# Path to the directory with known faces
KNOWN_FACES_DIR = "known_faces"
known_faces = []
known_names = []

# Load each known face
for filename in os.listdir(KNOWN_FACES_DIR):
    # Load image
    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
    # Get the face encodings (assuming there's only one face per image)
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    # Store the name (assuming filename is the person's name)
    known_names.append(os.path.splitext(filename)[0])

# Path to the video file
VIDEO_FILE = "FACEE.mp4"

# Open the video file
video_capture = cv2.VideoCapture(VIDEO_FILE)

# Check if the video file was opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    # Capture a single frame of video
    ret, frame = video_capture.read()

    # Check if frame was read successfully
    if not ret:
        print("Error: Unable to read frame.")
        break
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB
    
    # Find all face locations and face encodings in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # Loop over each face found in the frame
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # See if the face matches any known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]
        
        # Draw a box around the face
        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()
