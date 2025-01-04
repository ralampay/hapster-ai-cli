import os
import sys
import cv2
from ultralytics import YOLO

class VideoDetect:
    def __init__(self, model_file, video_file):
        self.model_file = model_file
        self.video_file = video_file


    def execute(self):
        cap = cv2.VideoCapture(self.video_file)

        if not cap.isOpened():
            print(f"Can't open: {self.video_file}")
            exit()

        # Get the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Frame rate: {fps} frames per second")

        print(f"Loading object detection model: {self.model_file}")
        model = YOLO(self.model_file)


        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = model(frame, conf=0.2)

            if len(results) > 0:
                for result in results:
                    print(f"xyxy: {result.boxes.xyxy}")
                    # Extract xyxy coordinates from the result
                    boxes = result.boxes.xyxy  # This will be a tensor of shape (n, 4), where n is the number of detections
                    
                    # Loop through each bounding box in the result
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)  # Convert the box coordinates to integers
                        
                        # Draw the bounding box on the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color with thickness 2

            # Convert back first for display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Display the frame in a window
            cv2.imshow(f"{self.video_file}", frame)

            # Wait for the key press with a delay corresponding to the frame rate
            # cv2.waitKey(1) means a 1 ms delay, so we adjust this based on the fps
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
