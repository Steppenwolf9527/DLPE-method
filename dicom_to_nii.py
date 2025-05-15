import os
import pydicom
import numpy as np
import nibabel as nib
from glob import glob

def dcm_series_to_nii(dcm_dir, output_path):
    """
    将DICOM序列转换为3D NIfTI文件，并顺时针旋转90度以更改方向

    参数:
        dcm_dir: 包含DICOM文件的目录路径
        output_path: 输出的.nii.gz文件路径
    """
    # 获取所有DICOM文件
    dcm_files = glob(os.path.join(dcm_dir, "*.dcm"))
    if not dcm_files:
        dcm_files = glob(os.path.join(dcm_dir, "*"))
        dcm_files = [f for f in dcm_files if not os.path.isdir(f)]

    if not dcm_files:
        raise ValueError(f"在目录 {dcm_dir} 中未找到DICOM文件")

    # 读取并排序DICOM文件
    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f)
            if hasattr(ds, 'SliceLocation'):
                slices.append(ds)
        except:
            continue

    if not slices:
        raise ValueError("无法读取有效的DICOM文件")

    # 按SliceLocation排序
    slices.sort(key=lambda x: x.SliceLocation)

    # 创建3D体积数据
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    volume = np.zeros(img_shape, dtype=slices[0].pixel_array.dtype)

    for i, s in enumerate(slices):
        volume[:, :, i] = s.pixel_array

    # 获取DICOM的空间信息
    pixel_spacing = [float(x) for x in slices[0].PixelSpacing]
    slice_thickness = float(slices[0].SliceThickness)
    image_orientation = [float(x) for x in slices[0].ImageOrientationPatient]
    image_position = [float(x) for x in slices[0].ImagePositionPatient]

    # 构建affine矩阵
    affine = np.eye(4)
    affine[:3, 0] = np.array(image_orientation[:3]) * pixel_spacing[0]
    affine[:3, 1] = np.array(image_orientation[3:]) * pixel_spacing[1]
    affine[:3, 2] = np.cross(affine[:3, 0], affine[:3, 1])
    affine[:3, 2] = affine[:3, 2] / np.linalg.norm(affine[:3, 2]) * slice_thickness
    affine[:3, 3] = image_position

    # 顺时针旋转 volume（在 x-y 平面）
    volume_rot0 = np.rot90(volume, k=-1, axes=(0, 1))
    volume_rot1 = np.rot90(volume_rot0, k=-1, axes=(0, 1))
    volume_rot2= np.rot90(volume_rot1, k=-1, axes=(0, 1))
    volume_rot = np.rot90(volume_rot2, k=-1, axes=(0, 1))
    # 修改 affine：对前两列进行顺时针旋转
    R = np.array([[0, 1],
                  [-1, 0]])
    affine[:3, :2] = affine[:3, :2] @ R

    # 创建并保存 NIfTI 图像
    nii_img = nib.Nifti1Image(volume_rot, affine)
    nib.save(nii_img, output_path)
    print(f"成功保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    dcm_directory = r"D:\桌面\毕设\数据集\example_data\COVID-19 inpatient"
    output_nii = r"D:\桌面\毕设\数据集\example_data\COVID-19_inpatient5.nii.gz"

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_nii), exist_ok=True)

    # 执行转换
    dcm_series_to_nii(dcm_directory, output_nii)
