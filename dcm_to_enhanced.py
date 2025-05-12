# 导入所需的模块和函数
import post_processing.parenchyma_enhancement as enhancement  # 导入肺实质增强的后处理模块
import prediction.predict_rescaled as predictor  # 导入经过缩放处理的预测模块
import Format_convert.dcm_np_converter as normalize  # 导入DICOM转换为numpy数组的模块
import Interface.visualize_example as visualize  # 导入可视化示例模块
import Tool_Functions.Functions as Functions  # 导入工具函数模块

"""
每个胸部CT在V100 GPU上需要大约1分钟进行增强处理。
"""

# 设置批处理大小，约需要 3GB 的 GPU 内存
batch_size = 4

# 定义训练模型的目录（可以根据需要进行调整）
trained_model_top_dict = '/root/DLPE-method/root/DLPE-method/trained_models/'
# 该目录下保存着训练好的模型

# 定义 DICOM 或 DCM 文件所在的目录（包含一个胸部CT扫描）
dcm_directory = '/root/DLPE-method/root/DLPE-method/example_data/input/'
# 该目录包含待处理的CT数据文件

# 设置增强后CT数据的输出目录
enhance_array_output_directory = '/root/DLPE-method/root/DLPE-method/example_output/'
# 保存增强后的CT数据到该目录

# 设置预测器的检查点目录
predictor.top_directory_check_point = trained_model_top_dict

# 将 DICOM 文件转换为重采样后的CT数组，并应用合适的窗位和窗宽（可根据需要设置为 None）
rescaled_ct_array = normalize.dcm_to_spatial_signal_rescaled(dcm_directory, wc_ww=(-600, 1600))
# 如果 DICOM 文件本身包含正确的肺窗信息，wc_ww 可以设置为 None

# 预测肺部掩码（不进行精细化处理）
lung_mask = predictor.predict_lung_masks_rescaled_array(rescaled_ct_array, refine=False, batch_size=batch_size)

# 获取气道掩码（不进行精细化处理）
airway_mask = predictor.get_prediction_airway(rescaled_ct_array, lung_mask=lung_mask, semantic_ratio=0.02,
                                              refine_airway=False, batch_size=batch_size)
# refine_airway 会保留一个连通组件（丢弃约1%的预测正样本），处理时间约30秒

# 获取血管掩码（不进行精细化处理）
blood_vessel_mask = predictor.get_prediction_blood_vessel(rescaled_ct_array, lung_mask=lung_mask, semantic_ratio=0.1,
                                                          refine_blood_vessel=False, batch_size=batch_size)
# refine_blood_vessel 会保留一个连通组件（丢弃约2%的预测正样本），处理时间约60秒

# 使用全局采样方法去除气道和血管，并进行增强处理
DLPE_enhanced, w_l, w_w = enhancement.remove_airway_and_blood_vessel_general_sampling(rescaled_ct_array, lung_mask,
                                                                                      airway_mask, blood_vessel_mask, window=True)
# 得到增强后的CT图像，以及推荐的窗位和窗宽
w_l = round(w_l)  # 用于观察亚视觉肺实质病变的最佳窗位
w_w = round(w_w)  # 用于观察亚视觉肺实质病变的窗宽
print("\n\n#############################################")
print("the scan level optimal window level is:", w_l, "(HU)")  # 输出最佳窗位
print("recommend window width is:", w_w, "(HU)")  # 输出推荐的窗宽
print("#############################################\n\n")

# 可视化增强后的CT图像的某一切片
example_slice = visualize.generate_slice(rescaled_ct_array, DLPE_enhanced, airway_mask, blood_vessel_mask,
                                         slice_z=250, show=True)

# 保存生成的图像，使用高分辨率
Functions.image_save(example_slice, enhance_array_output_directory + 'slice image name.png', high_resolution=True)

# 保存增强后的CT数据（以压缩格式）
Functions.save_np_array(enhance_array_output_directory, 'enhanced_ct_name', DLPE_enhanced, compress=True)

#######################################################################
# 以下是COVID-19幸存者的亚视觉病变分割。
# 请注意，该分割模型仅适用于COVID-19病变。
#######################################################################

# 导入增强后的预测模块，用于对增强后的CT进行COVID-19病变分割
import prediction.predict_enhanced as predict_enhanced
predict_enhanced.top_directory_check_point = trained_model_top_dict

# 获取后续随访患者的可见和亚视觉COVID-19病变掩码
follow_up_lesions = predict_enhanced.get_invisible_covid_19_lesion_from_enhanced(
    DLPE_enhanced, follow_up=True, batch_size=batch_size)

# 获取住院患者的可见和亚视觉COVID-19病变掩码
inpatient_lesions = predict_enhanced.get_invisible_covid_19_lesion_from_enhanced(DLPE_enhanced, follow_up=False,
                                                                                 batch_size=batch_size)

# 获取COVID-19感染的可见病变掩码（基于论文TMI 2020年，DOI: 10.1109/TMI.2020.3001810）
visible_lesions = predictor.predict_covid_19_infection_rescaled_array(rescaled_ct_array, lung_mask=lung_mask,
                                                                      batch_size=batch_size)
