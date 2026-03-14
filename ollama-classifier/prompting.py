system_prompt = """
You are a neurologist specializing in epilepsy reading and annotating electroencephelogram (EEG) reports. 
Use these STRICT rules to return key findings:

- DO NOT classify based on text from the report's "HISTORY:" or "Relevant History:" sections (case-insensitive).
- Primarily use the "Findings:", "Impressions:", and "Summary:" sections to label the report (case-insensitive). 
- Return ONLY valid JSON (no markdown, no extra text).
"""


evidence = "If value is False return null; ONLY provide if value is True: a short, verbatim exerpt of the evidence used to label."


study_no = "Return the alpha-numerial sequence, usually hyphenated (sometimes without alphabet characters). Example: 17-789C"
emu = "If recording occurred in the Epilepsy Monitoring Unit (EMU), denote its Phase."

diff_enc = "True for generalized, widespread patterns of slowed brain waves, usually mentioned as 'generalized background slowing' or 'diffuse nonspecific abnormalities'"
mod = "Return the modifier of severity that preceds the declaration of diffuse encephalopathy."

e_sz = "True if a seizure occurred and was eileptic or electrographic (had an EEG correlate)."
f_sz = "True if a seizures occured and was non-epileptic or functional (did not have an EEG correlate)."
se = "True ONLY for explicit mentions of 'status epilepticus'."
ncse = "True ONLY for explicit mentions of 'nonconvulsive status epilepticus.'"

ed = "Denote occurence of epileptiform discharges."
ed_type = "Return all occuring epileptiform discharge types."


asym = "True for mentions of asymmetric background activity or asymmetric sleep activity."
iic = "True ONLY for explicit phrases like 'ictal-interictal-continuum', or abbreviated 'IIC'."
lpd = "True ONLY for explicit phrases like 'lateral periodic discharges', or abbreviated 'LPDs. Generalized periodic discharges (GPDs) => False."
gpd = "True ONLY for explicit mentions of 'generalized periodic discharges' or 'GPDs'. DO NOT include if 'with triphasic morphology'."
tw = "True ONLY for explicit mentions of 'generalized periodic discharges with triphasic morphology', or 'GPDs with triphasic morphology', 'triphasic waves', or 'triphasiform potentials'."
fs = "True ONLY for explicit mentions of 'focal slowing' or 'focal dysfunction'."
