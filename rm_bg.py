from rembg import remove
from PIL import Image
import time

# Store path of the image in the variable input_path
input_path = 'datasets/mot/train/MOT17-02-FRCNN/img1/000521.jpg'

# Store path of the output image in the variable output_path
output_path = 'output.png'

# Open the image
with open(input_path, 'rb') as img:
    input_img = img.read()

start_time = time.time()
# Remove the background from the image
output_img = remove(input_img)
end_time = time.time()
print('Time taken:', end_time - start_time)

# Save the image with the background removed
with open(output_path, 'wb') as out:
    out.write(output_img)
