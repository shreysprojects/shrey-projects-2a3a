"""Generate a sample output file for every pair in translation_inputs.json.

Uses whatever provider MODEL_PROVIDER points to (default 'mock', which needs no
key). Writes one JSON file per case into expected_outputs/translation/. Handy
for demoing the pipeline and for regenerating samples after a change.

Run from the project root:
    python samples/generate_translation_outputs.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:  # load .env so MODEL_PROVIDER / API keys are picked up (same as the CLI)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.translation_quality_checker import (  # noqa: E402
    TranslationError,
    check_translation,
)

HERE = pathlib.Path(__file__).resolve().parent
OUTDIR = HERE / "expected_outputs" / "translation"


def main() -> None:
    inputs = json.loads((HERE / "translation_inputs.json").read_text(encoding="utf-8"))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for item in inputs:
        cid = item["id"]
        try:
            result = check_translation(
                item["english_original"], item["spanish_translation"]
            )
            payload = json.loads(result.model_dump_json())
        except TranslationError as exc:
            payload = {"error": str(exc)}
        (OUTDIR / f"case_{cid}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"wrote expected_outputs/translation/case_{cid}.json  ({item['case']})")


if __name__ == "__main__":
    main()
