from PIL import Image

# Open an image file
with Image.open('/home/sonnguyen112/Pictures/6045d8bd0572a82cf163.jpg') as img:
    # Calculate target size maintaining aspect ratio 13:18
    width = img.size[0]
    height = img.size[1]
    aspect_ratio = width / height

    target_aspect_ratio = 13 / 18
    if aspect_ratio > target_aspect_ratio:
        # If image is wider than the target aspect ratio, constrain width and adjust height
        target_width = width
        target_height = round(target_width / target_aspect_ratio)
    else:
        # If image is taller than the target aspect ratio, constrain height and adjust width
        target_height = height
        target_width = round(target_height * target_aspect_ratio)

    # Resize the image
    img_resized = img.resize((target_width, target_height))

# Save the resized image
img_resized.save('/home/sonnguyen112/Desktop/output.png')
