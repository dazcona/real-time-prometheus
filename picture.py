# Imports
import os
import requests
from visualize import visualize_detections
from matplotlib.pyplot import imsave

# Padding
pad_width, pad_height = 850., 800

# API
SERVER = 'localhost'
PORT = 4000
API_URI = 'http://%s:%s/' % (SERVER, PORT)

# Image Path
path = 'argentina'
filename = '1.jpeg'
filepath = os.path.join(path, filename)

# Image Data
from PIL import Image
from io import BytesIO
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
        imsave(os.path.join(path, 'tagged.jpeg'), img_tagged)
