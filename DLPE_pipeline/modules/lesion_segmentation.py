from prediction import predict_enhanced
from DLPE_pipeline.config import TRAINED_MODEL_DIR, BATCH_SIZE,OUTPUT_ROOT
import nibabel as nib
import numpy as np
import os
def run(context: 'PipelineContext'):
    predict_enhanced.top_directory_check_point = TRAINED_MODEL_DIR

    context.inpatient_lesions = predict_enhanced.get_invisible_covid_19_lesion_from_enhanced(
        context.DLPE_enhanced, follow_up=False, batch_size=BATCH_SIZE
    )

    context.follow_up_lesions = predict_enhanced.get_invisible_covid_19_lesion_from_enhanced(
        context.DLPE_enhanced, follow_up=True, batch_size=BATCH_SIZE
    )
    # -------- 合并掩膜为多标签图 --------
    # 标签定义：
    # 0 = 背景
    # 1 = 肺部
    # 2 = 感染区域（在肺部基础上进一步覆盖）

    # 转换数据类型为 uint8（节省空间 + 适用于 NIfTI）
    combined_mask = context.lung_mask.astype(np.uint8)

    # 将炎症区域设为标签 2（覆盖肺部标签）
    combined_mask[context.inpatient_lesions == 1] = 2

    print("✅ The automatic annotation of inflammatory areas during the hospital stay has been completed.")
    print("✅ Generating combined multi-label lung mask...")

    # -------- 导出为 .nii.gz --------


    affine = np.eye(4)  # 或替换为原始 CT 图像 affine

    # 创建 NIfTI 图像
    nii_img = nib.Nifti1Image(combined_mask, affine)

    # 输出路径，使用增强输出目录
    # output_dir = os.path.join(OUTPUT_ROOT, context.patient_id)
    # os.makedirs(output_dir, exist_ok=True)
    output_dir = OUTPUT_ROOT

    output_path = os.path.join(output_dir, 'combined_lung_infection_mask.nii.gz')
    nib.save(nii_img, output_path)

    print(f"✅ Combined multi-label mask saved as: {output_path}")
