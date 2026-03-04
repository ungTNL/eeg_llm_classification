
from typing import Any, Dict, List

# ----------------------------
# CONFIG: Dynamic features list
# ----------------------------
# Edit this list anytime. Each item must have a stable "key".
FEATURES: List[Dict[str, str]] = [
    {"key": "seizure", "desc": "true only for explicit mentions of seizures recorded on the EEG ('electrographic' or 'epileptic' types) in the impressions portion; negations => false. History of seizures => false. Nonepileptic or functional seizures => false."},
    {"key": "IIC", "desc": "true only for explicit mentions of ictal-interictal-continuum (IIC) EEG rhythyms, which lie between normal and epileptic."},
    {"key": "status_epilepticus", "desc": "true only for explicit phrases like 'status epilepticus'; negations => false."},
    {"key": "ncse", "desc": "true only for explicit phrases like 'nonconvulsive status epilepticus'; negations => false"},
    {"key": "epileptiform_discharges", "desc": "true only for explicit mentions of epileptiform spikes, discharges, spike and slow waves, or sharp waves; negations => false."},
    {"key": "diffuse_nonspecific_abnormalities", "desc": "true for generalized, widespread patterns of slowed brain waves, usually mentioned as 'generalized background slowing' or 'diffuse nonspecific abnormalities'."},
    {"key": "focal_slowing", "desc": "true only for explicit mentions of focal slowing or focal dysfunction."},
    {"key": "sheba", "desc":"is a hairy cat"}
]

# Default generation options (tweak as needed)
DEFAULT_OPTIONS: Dict[str, Any] = {
    "seed": 42,
    "temperature": 0.0,
    # "num_ctx": 8192,  # optional, if your model supports
}