
from typing import Any, Dict, List

# ----------------------------
# CONFIG: Dynamic features list
# ----------------------------
# Edit this list anytime. Each item must have a stable "key".
FEATURES: List[Dict[str, str]] = [
    {"key": "asym",
        "opts": ['True', 'False'],
        "desc": "True for mentions of asymmetric background activity or asymmetric sleep activity."},
    {"key": "BiRD",
        "opts": ['True', 'False'],
        "desc": "True ONLY for explicit phrases like 'brief potentially ictal rhythmic discharges', or 'BIRDs'."},
    {"key": "eSZ",
        "opts": ['True', 'False'],
        "desc": "True ONLY for explicit mentions of seizures with an EEG correlate (electrographic or epileptic types) in the impressions portion. Negations => False. History of seizures => False. Nonepileptic or functional seizures => False."},
    {"key": "fSZ",
        "opts": ['True', 'False'],
        "desc": "True ONLY for explicit mentions of seizures WITHOUT an EEG correlate (functional or nonepileptic types). Electrographic or epileptic seizures => False."},   
    {"key": "SE",
        "opts": ['True', 'False'],
        "desc": "True ONLY for explicit phrases like 'status epilepticus'."},
    {"key": "ncSE",
        "opts": ['True', 'False'],
        "desc": "True ONLY for explicit phrases like 'nonconvulsive status epilepticus'."},
    {"key": "IIC",
        "opts": ['True', 'False'],
        "desc": "True ONLY for explicit phrases like 'ictal-interictal-continuum', or abbreviated 'IIC'."},
    {"key": "GPD",
        "opts": ['True', 'False'],
        "desc": "True ONLY for explicit mentions of 'generalized periodic discharges' or 'GPDs'. DO NOT include if 'with triphasic morphology'."},
    {"key": "GPDt",
        "opts": ['True', 'False'], 
        "desc": "True ONLY for explicit mentions of 'generalized periodic discharges with triphasic morphology', or 'GPDs with triphasic morphology', 'triphasic waves', or 'triphasiform potentials'."},
    {"key": "LPD",
        "opts": ['True', 'False'],
        "desc": "True ONLY for explicit phrases like 'lateral periodic discharges', or abbreviated 'LPDs. Generalized periodic discharges (GPDs) => False."},
    {"key": "fs",
        "opts": ['True', 'False'], 
        "desc": "True ONLY for explicit mentions of focal slowing or focal dysfunction."},
    {"key": "focRDA",
        "opts": ['True', 'False'],
        "desc": "True ONLY for focalized rhythmic delta activity, phrases like 'lateral rhythmic delta activity', 'LRDA', 'temporal intermittent delta activity', 'TIRDA'. GRDA => False. Generalized rhythmic delta activity => False."},
    {"key": "fpf",
        "opts": ['True', 'False'],
        "desc": "True ONLY for explicit phrasing of 'focal paroxsymal fast'. Generalized paroxsymal fast => False."},
    {"key": "gpf",
        "opts": ['True', 'False'], 
        "desc": "True ONLY for explicit phrasing of 'generalized paroxsymal fast'. Focal paroxsymal fast => False."},
    {"key": "GRDA",
        "opts": ['True', 'False'], 
        "desc": "True ONLY explicit mentions of generalized ryhthmic delta activity (GRDA). Lateral rhythmic delta activity (LRDA) => False. Temporal intermittent rhythmic delta activity (TiRDA) => False."},
    {"key": "ha",
        "opts": ['True', 'False'], 
        "desc": "True ONLY for explicit mentions of 'hypsarrythmia'."},
    {"key": "SW",
        "opts": ['True', 'False'], 
        "desc": "True ONLY for explicit mentions of spike and wave, polyspike and wave, sharp and wave, epileptiform spikes or sharps."},
    {"key": "dna",
        "opts": ['True', 'False'], 
        "desc": "True for generalized, widespread patterns of slowed brain waves, usually mentioned as 'generalized background slowing' or 'diffuse nonspecific abnormalities'."}
]


# Default generation options (tweak as needed)
DEFAULT_OPTIONS: Dict[str, Any] = {
    "seed": 42,
    "temperature": 0.0,
    # "num_ctx": 16384,  # use export OLLAMA_CONTEXT_LENGTH=16384
    # also do: export OLLAMA_FLASH_ATTENTION=1
    # optionally: export OLLAMA_KV_CACHE_TYPE=q8_0
}