from DLPE_pipeline.config import TRAINED_MODEL_DIR, BATCH_SIZE, SHOW_SLICE, SLICE_Z, OUTPUT_ROOT
from Format_convert.dcm_np_converter import dcm_to_spatial_signal_rescaled
from prediction import predict_rescaled
from post_processing import parenchyma_enhancement
from Interface import visualize_example
from Tool_Functions import Functions
import os
import numpy as np
import nibabel as nib
def run(context: 'PipelineContext', dcm_directory: str):
    predict_rescaled.top_directory_check_point = TRAINED_MODEL_DIR

    context.rescaled_ct_array = dcm_to_spatial_signal_rescaled(dcm_directory, wc_ww=(-600, 1600))

    context.lung_mask = predict_rescaled.predict_lung_masks_rescaled_array(
        context.rescaled_ct_array, refine=False, batch_size=BATCH_SIZE
    )

    context.airway_mask = predict_rescaled.get_prediction_airway(
        context.rescaled_ct_array, context.lung_mask,
        semantic_ratio=0.02, refine_airway=False, batch_size=BATCH_SIZE
    )

    context.blood_vessel_mask = predict_rescaled.get_prediction_blood_vessel(
        context.rescaled_ct_array, context.lung_mask,
        semantic_ratio=0.1, refine_blood_vessel=False, batch_size=BATCH_SIZE
    )

    context.DLPE_enhanced, w_l, w_w = parenchyma_enhancement.remove_airway_and_blood_vessel_general_sampling(
        context.rescaled_ct_array, context.lung_mask, context.airway_mask, context.blood_vessel_mask, window=True
    )

    # 创建仿射矩阵（CT 一般为单位矩阵）
    affine = np.eye(4)

    # 创建 NIfTI 图像
    nii_img_ct = nib.Nifti1Image(context.DLPE_enhanced.astype(np.float32), affine)

    # 生成保存路径
    ct_save_path = os.path.join(OUTPUT_ROOT, 'enhanced_ct_image.nii.gz')
    # ct_save_path = os.path.join(OUTPUT_ROOT, context.patient_id, 'enhanced_ct_image.nii.gz')

    # 保存为 .nii.gz
    nib.save(nii_img_ct, ct_save_path)

    print(f" The enhanced CT image is saved as: {ct_save_path}")
    if SHOW_SLICE:
        slice_img = visualize_example.generate_slice(
            context.rescaled_ct_array, context.DLPE_enhanced,
            context.airway_mask, context.blood_vessel_mask,
            slice_z=SLICE_Z, show=True
        )
        Functions.image_save(slice_img, os.path.join(OUTPUT_ROOT, 'enhanced_slice.png'))
        # Functions.image_save(slice_img, os.path.join(OUTPUT_ROOT, context.patient_id, 'enhanced_slice.png'))

    # Functions.save_np_array(os.path.join(OUTPUT_ROOT, context.patient_id), 'enhanced_ct', context.DLPE_enhanced)

