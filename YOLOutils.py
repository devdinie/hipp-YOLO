from __future__ import annotations
import os
import numpy as np
import SimpleITK as sitk

def get_annotations(bbox, msk_nii, mskFile_name, save_annot=True):
	[x1, y1, z1, x2, y2, z2] = bbox

	x_bb_ctr = (x1+x2)/2
	y_bb_ctr = (y1+y2)/2
	z_bb_ctr = (z1+z2)/2

	x_bb_wr = round((x2-x1)/msk_nii.GetSize()[0],2)
	y_bb_wr = round((y2-y1)/msk_nii.GetSize()[1],2)
	z_bb_wr = round((z2-z1)/msk_nii.GetSize()[2],2)

	msk_arr = sitk.GetArrayFromImage(msk_nii)
	labels  = np.nonzero(np.unique(msk_arr))

	annotations = np.append([mskFile_name, x_bb_ctr, y_bb_ctr, z_bb_ctr, x_bb_wr, y_bb_wr, z_bb_wr], labels)
	
	if save_annot:
		writeto_txtFile(annotations, labels)
	else:
		return [x_bb_ctr, y_bb_ctr, z_bb_ctr, x_bb_wr, y_bb_wr, z_bb_wr]

def writeto_txtFile(annotations,labels):
	for label in labels:
		print(label)
	#File = open("classes.txt", 'a')
	#labels_str = np.array2string(labels)
	#File.write(labels_str+ "\n")

	File = open("annotations.txt", 'a')
	annot_str = np.array2string(annotations, precision=2, separator=',').replace("'","").strip("[]")
	File.write(annot_str+ "\n")

	
