import Tool_Functions.visualize_mask_mesh as Mesh
import os
from DLPE_pipeline.config import OUTPUT_ROOT

def run(context: 'PipelineContext'):
    output_path = os.path.join(OUTPUT_ROOT, "combined_lung_model.glb")
    Mesh.generate_combined_mask_mesh(
        lung_mask=context.lung_mask,
        airway_mask=context.airway_mask,
        vessel_mask=context.blood_vessel_mask,
        output_path=output_path
    )
