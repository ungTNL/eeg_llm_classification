import ollama
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
from pydantic import BaseModel, ValidationError

# custom modules
from prompting import *
from utilities import extract_json_object


# ----------------------------
# Pydantic models (labels only)
# ----------------------------

class NoteLabels(BaseModel):
    # dynamic: map feature_key -> true/false/null
    features: Dict[str, bool]

SCHEMA = NoteLabels.model_json_schema()

# ----------------------------
# Ollama call + retry
# ----------------------------

def call_ollama_chat(
    client: ollama.Client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    options: Dict[str, Any],
) -> str:
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # If your version errors on this kwarg, remove it:
        format=SCHEMA,
        options=options,
    )
    try:
        return resp["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected ollama response shape: {type(resp)} {resp}") from e


def infer_one_note_with_retries(
    client: ollama.Client,
    model: str,
    note_id: str,
    note_text: str,
    features: List[Dict[str, str]],
    options: Dict[str, Any],
    max_retries: int,
    base_backoff_s: float,
    wallclock_timeout_s: int,
) -> Tuple[NoteLabels, str]:
    system_prompt = build_system_prompt(features)
    feature_keys = [f["key"] for f in features]
    user_prompt = build_user_prompt(note_id, note_text, feature_keys)

    t0 = time.time()
    last_err: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        if (time.time() - t0) > wallclock_timeout_s:
            raise TimeoutError(f"Wallclock timeout exceeded ({wallclock_timeout_s}s) for note_id={note_id}")

        try:
            raw = call_ollama_chat(client, model, system_prompt, user_prompt, options)
            obj = extract_json_object(raw)

            # Force note_id (avoid occasional model drift)
            obj["note_id"] = str(note_id)

            parsed = NoteLabels.model_validate(obj)

            # Ensure all requested keys exist; fill missing with null
            for k in feature_keys:
                if k not in parsed.features:
                    parsed.features[k] = None

            # Optionally drop any extra keys the model invented
            parsed.features = {k: parsed.features.get(k, None) for k in feature_keys}

            # Ensure values are strictly bool/None (guard against "yes"/"no")
            for k, v in parsed.features.items():
                if v is None or isinstance(v, bool):
                    continue
                raise ValidationError(f"Feature {k} must be bool or null; got {type(v)}", NoteLabels)

            return parsed, raw

        except Exception as e:
            last_err = e
            logging.warning("Attempt %d failed for note_id=%s: %s", attempt + 1, note_id, repr(e))
            if attempt >= max_retries:
                break
            time.sleep(base_backoff_s * (2 ** attempt))

    assert last_err is not None
    raise last_err
