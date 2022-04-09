import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D


img_arr = nib.load('../CP01_004-n3-fli-resize.nii.gz').get_data()
print(img_arr.shape)

OrthoSlicer3D(img_arr).show()