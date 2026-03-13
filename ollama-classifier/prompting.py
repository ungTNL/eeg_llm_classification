from typing import List, Dict

# ----------------------------
# Prompting
# ----------------------------

def build_system_prompt(features: List[Dict[str, str]]) -> str:
    feat_lines = "\n".join([f'- "{f["key"]}": {f["desc"]}' for f in features])
    return f"""
Task:
You are a neurologist specializing in epilepsy reading and annotating electroencephelogram (EEG) reports. 
Use the following annotation guide to understand the terminology and return key findings:

{feat_lines}

Output rules (STRICT):
- DO NOT classify based on text from the report's "HISTORY:" or "Relevant History:" sections (case-insensitive).
- Primarily use the "Findings:", "Impressions:", and "Summary:" sections to label the report (case-insensitive). 
- Return ONLY valid JSON (no markdown, no extra text).
- Return ONLY 1 label (True | False).
- Include every requested feature key exactly once in "features".

JSON schema:
{{
  "features": {{
    "<feature_key>": True | False,
    ...
  }}
}}
"""


def build_user_prompt(note_text: str, feature_keys: List[str]) -> str:
    keys = ", ".join(feature_keys)
    return f"""Requested features: {keys}

EEG report to annotate:
\"\"\"{note_text}\"\"\"
"""