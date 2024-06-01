# # list all file of folder
# import os
# import sys
# import shutil

# files = os.listdir("dataset/mix_det/train/labels")
# print(len(files))

# for file in files:
#     new_lines = []
#     with open("dataset/mix_det/train/labels/" + file, "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip()
#             label = line.split(" ")
#             cls, x, y, w, h = label
#             cls = int(cls)
#             new_line = " ".join([str(cls + 1), str(x), str(y), str(w), str(h)])
#             print(new_line)
#             new_lines.append(new_line)
#     with open("dataset/mix_det/train/labels/" + file, "w") as f:
#         for line in new_lines:
#             f.write(line + "\n")
