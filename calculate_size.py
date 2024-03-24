import os

def calculate_image_size(directory, extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                fp = os.path.join(dirpath, filename)
                total_size += os.path.getsize(fp)
    return total_size

# Usage
directory = 'datasets'  # replace with your directory
# print(list(os.walk(directory)))
total_size = calculate_image_size(directory)
print(f'Total image size in "{directory}" is {total_size / 1024 / 1024 / 1024} GB.')
