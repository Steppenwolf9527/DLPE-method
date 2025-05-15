import numpy as np
import nibabel as nib
import os
import sys
import traceback

def npz_to_nii(npz_path, output_path):
    if not os.path.exists(npz_path):
        print(f"错误：文件 {npz_path} 不存在")
        return

    try:
        # 使用 memory map 读取，防止内存爆炸
        npz_data = np.load(npz_path, mmap_mode='r')
    except Exception as e:
        print(f"无法读取 npz 文件: {e}")
        return

    array = None
    for key in npz_data.files:
        array = npz_data[key]
        print(f"使用数据键: {key}, 数据形状: {array.shape}, 数据类型: {array.dtype}")
        break

    if array is None:
        print("错误：npz 文件中没有可用的数据数组")
        return

    affine = np.eye(4)

    try:
        # 不做任何转换，直接保存为 NIfTI
        nii_image = nib.Nifti1Image(array, affine)
        nib.save(nii_image, output_path)
        print(f"保存成功: {output_path}")
    except Exception as e:
        print("保存失败:")
        traceback.print_exc()

if len(sys.argv) != 3:
    print("用法: python npz_to_nii.py <输入.npz文件路径> <输出.nii.gz文件名>")
else:
    npz_to_nii(sys.argv[1], sys.argv[2])
