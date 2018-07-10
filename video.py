# Imports
import cv2
import os
import requests
from visualize import visualize_detections
from matplotlib.pyplot import imsave
from PIL import Image
from io import BytesIO
import time

# Padding
pad_width, pad_height = 850., 800

# API
SERVER = 'localhost'
PORT = 4000
API_URI = 'http://%s:%s/' % (SERVER, PORT)

# Image Path
path = 'video_images'

# Video capture
cam = cv2.VideoCapture(0)
window_name = "Prometheus"
cv2.namedWindow(window_name)

while True:

    ret, frame = cam.read()
    cv2.imshow(window_name, frame)

    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:

        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k % 256 == 32:

        # SPACE pressed
        name = time.strftime("%Y%m%d-%H%M%S")
        filename = "frame_{}.png".format(name)
        filepath = os.path.join(path, filename)
        cv2.imwrite(filepath, frame)
        print("{} written!".format(filename))

        # Image Data
        with BytesIO() as output:
            with Image.open(filepath) as img:
                img.save(output, 'BMP')
            image_data = output.getvalue()

        # Dictionary of files
        files = {}
        files[filename] = (filename, image_data, 'image/jpeg')

        # Score
        print('Calling API to score image...')
        r = requests.post(API_URI + 'score', files=files)

        # Status
        print('Status code:', r.status_code)

        if r.status_code == 200:

            # RESULTS
            results = r.json()

            if len(results) > 0:
                # First result
                result = results[0]

                # Tagged
                img_tagged = visualize_detections(filepath,
                                                  result['boxes'],
                                                  result['labels'],
                                                  result['scores'],
                                                  pad_width, pad_height,
                                                  classes=[0, 1],
                                                  draw_negative_rois=False,
                                                  decision_threshold=0.75)
                # Save
                tagged_filename = "frame_{}_tagged.png".format(name)
                tagged_filepath = os.path.join(path, tagged_filename)
                imsave(tagged_filepath, img_tagged)
                # Show
                with Image.open(tagged_filepath) as img:
                    img.show()

cam.release()
cv2.destroyAllWindows()