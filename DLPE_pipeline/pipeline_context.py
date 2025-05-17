class PipelineContext:
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.rescaled_ct_array = None
        self.lung_mask = None
        self.airway_mask = None
        self.blood_vessel_mask = None
        self.DLPE_enhanced = None
        self.inpatient_lesions = None
        self.follow_up_lesions = None
