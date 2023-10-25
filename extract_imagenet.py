"""
    Convert tar file to zip file
    Author: Dingdong Yang
"""

import cv2
import numpy as np

import tarfile
import zipfile

IMAGE_SIZE = 256
tar_file_path = "ILSVRC2012_img_train.tar"
dest_zip_file_path = "imagenet_train.zip"
dest_zip_file = zipfile.ZipFile(dest_zip_file_path, mode="w")

def center_crop(read_out):
    # Read_out is the binary/strings
    nparr = np.fromstring(read_out, np.uint8) 
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_np = cv2.resize(img_np, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    # Encode to string and write to zip file
    img_str = cv2.imencode('.jpeg', img_np)[1].tobytes()
    return img_str


count = 0
top_tar = tarfile.open(tar_file_path)
file_list = top_tar.getnames()
for file in file_list:
    if file.endswith("tar"):
        sub_tar = tarfile.open(fileobj=top_tar.extractfile(file))
        sub_file_list = sub_tar.getnames()
        for sub_file in sub_file_list:
            # Read out bytes and convert to numpy images
            read_out = sub_tar.extractfile(sub_file).read()
            img_str = center_crop(read_out)
            dest_zip_file.writestr(sub_file, img_str)
            count += 1
        sub_tar.close()
    elif file.endswith(".JPEG"):
        read_out = top_tar.extractfile(file).read()
        img_str = center_crop(read_out)
        dest_zip_file.writestr(file, img_str)
        count += 1


print(f"Total {count} images are processed to {dest_zip_file_path}.")
top_tar.close()
dest_zip_file.close()
