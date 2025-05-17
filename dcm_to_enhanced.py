# 导入各模块：分别包含增强处理、预测、DICOM 转换、可视化及工具函数
import post_processing.parenchyma_enhancement as enhancement
import prediction.predict_rescaled as predictor
import Format_convert.dcm_np_converter as normalize
import Interface.visualize_example as visualize
import Tool_Functions.Functions as Functions
import Tool_Functions.generate_ct_analysis_report as Report
import Tool_Functions.visualize_mask_mesh as Mesh
import numpy as np
import nibabel as nib
import os

"""
每一例胸部 CT 增强处理约耗时一分钟，需使用 V100 GPU。
"""

print("开始执行")
batch_size = 4  # 推理时每次送入模型的批大小，约占 3GB GPU 显存

# 模型所在目录
trained_model_top_dict = '/home/featurize/DLPE-method/trained_models/'

# 当前处理的 CT 图像目录，仅包含一例 CT（多个 .dcm 文件）
dcm_directory = '/home/featurize/DLPE-method/example_data/COVID-19 inpatient/'

# 增强图像输出保存路径
enhance_array_output_directory = '/home/featurize/DLPE-method/example_output/'

# 设置模型加载的路径（用于 predict_rescaled 中调用模型）
predictor.top_directory_check_point = trained_model_top_dict

# -------------------------- 步骤一：图像读取与归一化 --------------------------

# 将 dcm 图像转换成统一空间分辨率并应用肺窗（WC: -600, WW: 1600），生成三维数组
rescaled_ct_array = normalize.dcm_to_spatial_signal_rescaled(dcm_directory, wc_ww=(-600, 1600))
# 如果 DICOM 中已有窗宽窗位信息，也可以设为 wc_ww=None 自动使用

# -------------------------- 步骤二：肺部分割 --------------------------

# 使用肺部分割模型提取肺区域 mask
lung_mask = predictor.predict_lung_masks_rescaled_array(
    rescaled_ct_array, refine=False, batch_size=batch_size
)

# -------------------------- 步骤三：气道提取 --------------------------

# 使用气道分割模型提取气道区域
airway_mask = predictor.get_prediction_airway(
    rescaled_ct_array,
    lung_mask=lung_mask,
    semantic_ratio=0.02,       # 占肺部比例估计（控制 mask 阈值）
    refine_airway=False,       # 是否只保留最大连通区域（精细化）
    batch_size=batch_size
)

# -------------------------- 步骤四：血管提取 --------------------------

# 使用血管分割模型提取肺内血管区域
blood_vessel_mask = predictor.get_prediction_blood_vessel(
    rescaled_ct_array,
    lung_mask=lung_mask,
    semantic_ratio=0.1,
    refine_blood_vessel=False,
    batch_size=batch_size
)

# -------------------------- 步骤五：肺实质增强 --------------------------
# 根据提取的肺部/气道/血管 mask，去除非肺实质结构，进行增强
DLPE_enhanced, w_l, w_w = enhancement.remove_airway_and_blood_vessel_general_sampling(
    rescaled_ct_array,
    lung_mask,
    airway_mask,
    blood_vessel_mask,
    window=True  # 自动计算推荐的窗宽窗位
)

# 将推荐的窗宽窗位四舍五入
w_l = round(w_l)
w_w = round(w_w)

# 打印推荐窗口参数
print("\n\n#############################################")
print("the scan level optimal window level is:", w_l, "(HU)")
print("recommend window width is:", w_w, "(HU)")
print("#############################################\n\n")

# -------------------------- 步骤六：切片可视化 --------------------------

# 可视化切片（显示原始图像、增强图、气道、血管），slice_z=250 表示第 250 层
example_slice = visualize.generate_slice(
    rescaled_ct_array,
    DLPE_enhanced,
    airway_mask,
    blood_vessel_mask,
    slice_z=250,
    show=True  # 显示窗口
)

# 保存该切片图像为高分辨率图
Functions.image_save(
    example_slice,
    enhance_array_output_directory + 'slice image name.png',
    high_resolution=True
)

# 保存增强后的三维 CT 数组（.npz 文件）
Functions.save_np_array(
    enhance_array_output_directory,
    'enhanced_ct_name',
    DLPE_enhanced,
    compress=True
)
# 创建仿射矩阵（CT 一般为单位矩阵）
affine = np.eye(4)

# 创建 NIfTI 图像
nii_img_ct = nib.Nifti1Image(DLPE_enhanced.astype(np.float32), affine)

# 生成保存路径
ct_save_path = os.path.join(enhance_array_output_directory, 'enhanced_ct_image.nii.gz')

# 保存

nib.save(nii_img_ct, ct_save_path)

print(f"：The enhanced CT image is saved as{ct_save_path}")
# Functions.save_np_as_nii_gz(
#     enhance_array_output_directory,
#     'enhanced_ct_name',
#     DLPE_enhanced
# )
# -------------------------- 三维重建--------------------------
Mesh.generate_combined_mask_mesh(
    lung_mask=lung_mask,
    airway_mask=airway_mask,
    vessel_mask=blood_vessel_mask,
    output_path=enhance_array_output_directory + "combined_lung_model.glb"
)
# -------------------------- 步骤七：COVID-19 病灶分割 --------------------------

print("For lung lesion segmentation in COVID-19 patients only")
# 仅适用于 COVID-19 病人的肺部病灶分割（注意：只适用于随访或住院人群）

# 加载增强模型权重路径
import prediction.predict_enhanced as predict_enhanced
predict_enhanced.top_directory_check_point = trained_model_top_dict

# 获取随访患者的可见 + 亚可见病灶 mask（模型专为 COVID-19 恢复期设计）
# 使用增强后的CT图像进行病灶分割
follow_up_lesions = predict_enhanced.get_invisible_covid_19_lesion_from_enhanced(
    DLPE_enhanced,
    follow_up=True,
    batch_size=batch_size
)

# 获取住院患者的可见 + 亚可见病灶 mask（模型参数略有不同）
inpatient_lesions = predict_enhanced.get_invisible_covid_19_lesion_from_enhanced(
    DLPE_enhanced,
    follow_up=False,
    batch_size=batch_size
)

Functions.save_np_array(
    enhance_array_output_directory,
    'inpatient_lesion_mask',
    inpatient_lesions,
    compress=True
)
# -------- 合并掩膜为多标签图 --------
# 标签定义：
# 0 = 背景
# 1 = 肺部
# 2 = 感染区域（在肺部基础上进一步覆盖）

# 转换数据类型为 uint8（节省空间 + 适用于 NIfTI）
combined_mask = lung_mask.astype(np.uint8)

# 将感染区域设为标签 2（覆盖肺部）
combined_mask[inpatient_lesions == 1] = 2
print("The automatic annotation of inflammatory areas during the hospital stay has been completed and saved as 'inpatient_lesion_mask.npz'.")
# -------- 导出为 .nii.gz --------
# 创建 affine（空间信息）为单位矩阵，或替换为原 CT 的 affine（如果有）
affine = np.eye(4)

# 创建 Nifti 图像对象
nii_img = nib.Nifti1Image(combined_mask, affine)

# 设置保存路径
output_dir = enhance_array_output_directory  # 你的保存目录
output_path = os.path.join(output_dir, 'combined_lung_infection_mask.nii.gz')

# 保存
nib.save(nii_img, output_path)
# -------------------------- 报告分析--------------------------
print(f" The merged mask has been saved as：{output_path}")
report = Report.generate_ct_analysis_report(
    dicom_dir=dcm_directory,
    lung_mask=lung_mask,
    lesion_mask=inpatient_lesions
)

print(report)
print("The automatic analysis report of inflammatory areas during the hospital stay has been completed.")
# 获取可见病灶（基于 TMI 论文中方法）
# 获取可见病灶
visible_lesions = predictor.predict_covid_19_infection_rescaled_array(
    rescaled_ct_array,
    lung_mask=lung_mask,
    batch_size=batch_size
)

