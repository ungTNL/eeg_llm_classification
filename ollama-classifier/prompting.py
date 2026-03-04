from typing import List, Dict

# ----------------------------
# Prompting
# ----------------------------

def build_system_prompt(features: List[Dict[str, str]]) -> str:
    feat_lines = "\n".join([f'- "{f["key"]}": {f["desc"]}' for f in features])
    return f"""
Task:
You are a neurologist specializing in epilepsy reading and annotating encephelography (EEG) reports. 
You must return what key findings are present. Use the following annotation guide to understand the terminology:
{feat_lines}

Output rules (STRICT):
- Return ONLY valid JSON (no markdown, no extra text).
- Return only 1 of the following 3 labels: "true", "false" or "null". Do not include additional text.
- If a feature is not mentioned or uncertain, output null.
- Include every requested feature key exactly once in "features".

JSON schema:
{{
  "features": {{
    "<feature_key>": true | false | null,
    ...
  }}
}}
"""


def build_user_prompt(note_id: str, note_text: str, feature_keys: List[str]) -> str:
    keys = ", ".join(feature_keys)
    return f"""note_id: {note_id}

Requested features: {keys}

EEG report to annotate:
\"\"\"{note_text}\"\"\"
"""