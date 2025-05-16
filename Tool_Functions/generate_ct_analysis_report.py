import pydicom
import os
import numpy as np


def extract_spacing_from_dicom(dicom_dir):
    """
    自动从 DICOM 文件夹中提取 spacing 信息。
    返回 (x_spacing, y_spacing, z_spacing)，单位：mm
    """
    files = sorted([os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
                    if f.endswith('.dcm') or f.endswith('.dicom')])
    if len(files) == 0:
        raise ValueError("DICOM 文件夹为空或格式不正确")

    # 读取第一张切片获取 x/y spacing
    ds = pydicom.dcmread(files[0])
    x_spacing, y_spacing = [float(v) for v in ds.PixelSpacing]

    # 读取至少两张切片计算 z spacing（slice 间距）
    if len(files) >= 2:
        ds1 = pydicom.dcmread(files[0])
        ds2 = pydicom.dcmread(files[1])
        z_spacing = abs(ds2.ImagePositionPatient[2] - ds1.ImagePositionPatient[2])
    else:
        # 有些 dicom 有 SliceThickness 字段
        z_spacing = float(getattr(ds, "SliceThickness", 1.0))

    return (x_spacing, y_spacing, z_spacing)


def generate_ct_analysis_report(dicom_dir, lung_mask, lesion_mask):
    """
    综合分析函数：提取 spacing，计算体积与占比，输出分析报告。
    """
    try:
        spacing = extract_spacing_from_dicom(dicom_dir)
        print(f"Successful automatic extraction of spacing.: {spacing} (单位 mm)")
    except Exception as e:
        print(f"⚠️ 无法自动提取 spacing，默认使用 (1.0, 1.0, 1.0): {e}")
        spacing = (1.0, 1.0, 1.0)

    # 体积计算
    voxel_volume_ml = np.prod(spacing) / 1000.0
    lesion_voxels = np.sum(lesion_mask == 1)
    lung_voxels = np.sum(lung_mask == 1)

    lesion_volume_ml = lesion_voxels * voxel_volume_ml
    lung_volume_ml = lung_voxels * voxel_volume_ml
    lesion_ratio = (lesion_volume_ml / lung_volume_ml) * 100 if lung_volume_ml > 0 else 0

    # 输出分析报告
    print("\n📊 ======= CT Lesion Analysis Report=======")
    print(f"Lesion Volume: {lesion_volume_ml:.2f} mL")
    print(f"Lung Volume: {lung_volume_ml:.2f} mL")
    print(f"Proportion of Lesion to Lung: {lesion_ratio:.2f} %")
    print("==================================\n")

    return {
        'lesion_volume_ml': lesion_volume_ml,
        'lung_volume_ml': lung_volume_ml,
        'lesion_ratio_percent': lesion_ratio,
        'spacing_mm': spacing
    }
