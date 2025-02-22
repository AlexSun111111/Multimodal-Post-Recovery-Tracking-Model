import os
import cv2
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from PIL import Image

file_root = r'model_out/test/'
save_path = r'model_out/test'
# file_list
file_list = os.listdir(file_root)
print(file_list)
for img_name in file_list:
    if img_name.endswith('.nii.gz'):
        img_path = file_root + img_name
        print(img_path)
        # data = np.load(img_path)
        # img1 = nib.load(img_path)
        # img = img1.get_fdata()
        img1 = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img1)
        img = (img - img.min()) / (img.max() - img.min())
        # print(img.min(),img.max())
        img = img * 255
        (x, y, z) = img.shape
        for i in range(x):
            data = img[i, :, :]
            path = save_path + "\\" + img_name.split('.')[0] + '\\' + str(i) + '.png'
            cv2.imwrite(path, data)

