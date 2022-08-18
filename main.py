## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# We want the points object to be persistent so we can display the
# last cloud when a frame drops
points = rs.points()
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a pipeline
pipeline = rs.pipeline()
# Create a config and configure the pipeline to stream
config = rs.config()
config.enable_stream(rs.stream.infrared)
# Start streaming
profile = pipeline.start(config)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        ir1_frame = frames.get_infrared_frame(1)  # Left IR Camera, it allows 1, 2 or no input
        image = np.asanyarray(ir1_frame.get_data())
        # Detect the faces
        faces = face_cascade.detectMultiScale(image, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.namedWindow('IR Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('IR Example', image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()