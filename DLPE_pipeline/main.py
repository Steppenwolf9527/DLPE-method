import argparse
import os
from config import DATA_ROOT
from pipeline_context import PipelineContext
from modules import enhancement, mesh_reconstruction, lesion_segmentation, report_analysis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', required=True)
    parser.add_argument('--patient_id', required=True)
    args = parser.parse_args()

    context = PipelineContext(args.patient_id)
    dcm_dir = os.path.join(DATA_ROOT, args.patient_id)

    if args.step == 'enhance':
        enhancement.run(context, dcm_dir)
    elif args.step == 'mesh':
        if context.lung_mask is None:
            raise RuntimeError("必须先执行增强步骤，未检测到 lung_mask")
        mesh_reconstruction.run(context)
    elif args.step == 'annotate':
        if context.DLPE_enhanced is None:
            raise RuntimeError("必须先执行增强步骤，未检测到 enhanced 图像")
        lesion_segmentation.run(context)
    elif args.step == 'report':
        if context.inpatient_lesions is None:
            raise RuntimeError("必须先执行病灶标注，未检测到 lesions")
        report_analysis.run(context, dcm_dir)
