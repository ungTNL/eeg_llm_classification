
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from prompting import *

# ----------------------------
# CONFIG: Dynamic features list
# ----------------------------
# Tunable parameter: 1 for local Mac, 10+ for HPC
CONCURRENCY_LIMIT: int = 1 
NUM_TEST: int = 2 # if wanting to test on a subset of the dataframe
MAX_RETRIES: int = 0


MODEL: str = "llama3.1"
MODEL_OPTIONS: Dict[str,Any] = {
    "num_ctx": 16384,
    "random_state": 42
}


# File I/O
INPUT_FILE: str = "../data/EEG_Notes_validation_set.xlsx"
DB_FILE: str = "eeg_results.db"
OUTPUT_FILE: str = "EEG_Notes_validation_set_classified.xlsx"

# Output Schema - PROMPTING
# --- OUTPUT SCHEMA - PROMPTING ---
class BoolResult(BaseModel):
    value: bool
    text: Optional[str] = Field(description=evidence, default=None, min_length=0, max_length=80)

    # MAGIC FIX: Intercept flat booleans and convert them to the proper dictionary format when LLM gets lazy!
    @model_validator(mode='before')
    @classmethod
    def unflatten_bool(cls, data: Any) -> Any:
        # Catch lazy flat booleans
        if isinstance(data, bool):
            return {"value": data, "text": None}
        
        # Catch LLM hallucinating an "assessment" wrapper
        if isinstance(data, dict):
            if 'assessment' in data:
                data = data['assessment']

            # Truncate text if the LLM gets overly verbose
            if 'text' in data and isinstance(data['text'], str):
                if len(data['text']) > 80:
                    data['text'] = data['text'][:80]
        
        return data

class Modifier(str, Enum):
    ABSENT = "None"
    MILD = "mild"
    MILD_MODERATE = "mild_moderate"
    MODERATE = "moderate"
    MODERATE_SEVERE = "moderate_severe"
    SEVERE = "severe"

class EDType(str, Enum):
    SHARPS = "sharps"
    SPIKES = "spikes"
    SPIKE_AND_SLOW = "spike and slow waves"
    SPIKE_AND_WAVE = "spike and waves"
    POLYSPIKES = "polyspikes"
    NONE = "None"

class EMULabel(str, Enum):
    PHASE_1 = "Phase 1"
    PHASE_2 = "Phase 2"
    PHASE_3 = "Phase 3"
    NONE = "None"

# NO MORE WRAPPER CLASSES - EVERYTHING IS FLATTENED!

class EEGExtraction(BaseModel):
    # ==========================================
    # STRATEGIC PLACEMENT: HIGH-ATTENTION FIELDS
    # ==========================================
    study_number: Optional[str] = Field(default=None, description=study_no, min_length=0, max_length=10)
    emu_label: Optional[EMULabel] = Field(default=None, description=emu, min_length=0, max_length=20)
    diffuse_encephalopathy_modifier: Optional[Modifier] = Field(default=None, description=mod)
    epileptiform_discharges_type: List[EDType] = Field(default_factory=list, description=ed_type)

    # ==========================================
    # STANDARD BOOLEAN FIELDS
    # ==========================================
    diffuse_encephalopathy: BoolResult = Field(description=diff_enc)
    sz_epileptic: BoolResult = Field(description=e_sz)
    sz_non_epileptic: BoolResult = Field(description=f_sz)
    sz_status_epilepticus: BoolResult = Field(description=se)
    sz_nonconvulsive_status_epilepticus: BoolResult = Field(description=ncse)
    epileptiform_discharges: BoolResult = Field(description=ed)
    asymmetric: BoolResult = Field(description=asym)
    ictal_interictal_cont: BoolResult = Field(description=iic)
    gen_pd: BoolResult = Field(description=gpd)
    gen_pdt: BoolResult = Field(description=tw)
    lat_pd: BoolResult = Field(description=lpd)
    focal_slowing: BoolResult = Field(description=fs)

    # MAGIC FIX: Ensure we never crash on a JSON 'null' if the LLM skips a key
    @model_validator(mode='before')
    @classmethod
    def fix_null_objects(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # A list of all our BoolResult fields
            bool_keys = [
                'diffuse_encephalopathy', 'sz_epileptic', 'sz_non_epileptic', 
                'sz_status_epilepticus', 'sz_nonconvulsive_status_epilepticus', 
                'epileptiform_discharges', 'asymmetric', 'ictal_interictal_cont', 
                'gen_pd', 'gen_pdt', 'lat_pd', 'focal_slowing'
            ]
            for key in bool_keys:
                if data.get(key) is None:
                    data[key] = {"value": False, "text": None}
        return data