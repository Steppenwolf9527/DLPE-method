import Tool_Functions.generate_ct_analysis_report as Report
from DLPE_pipeline.config import OUTPUT_ROOT
import os

def run(context: 'PipelineContext', dcm_directory: str):
    merged_output_path = os.path.join(OUTPUT_ROOT, context.patient_id, "merged_mask.nii.gz")
    report = Report.generate_ct_analysis_report(
        dicom_dir=dcm_directory,
        lung_mask=context.lung_mask,
        lesion_mask=context.inpatient_lesions
    )
    print(report)
    print("自动分析报告生成完成")
