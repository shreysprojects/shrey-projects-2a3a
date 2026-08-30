"""Generate a sample output file for every input in service_request_inputs.json.

Uses whatever provider MODEL_PROVIDER points to (default 'mock', which needs no
key). Writes one JSON file per case into expected_outputs/. Handy for demoing
the pipeline and for regenerating samples after a change.

Run from the project root:
    python samples/generate_sample_outputs.py
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

from src.service_request_cleanup import EmptyInputError, clean_service_request  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUTDIR = HERE / "expected_outputs"


def main() -> None:
    inputs = json.loads((HERE / "service_request_inputs.json").read_text(encoding="utf-8"))
    OUTDIR.mkdir(exist_ok=True)
    for item in inputs:
        cid = item["id"]
        try:
            result = clean_service_request(item["text"])
            payload = json.loads(result.model_dump_json())
        except EmptyInputError as exc:
            payload = {"error": str(exc)}
        (OUTDIR / f"case_{cid}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"wrote expected_outputs/case_{cid}.json  ({item['case']})")


if __name__ == "__main__":
    main()
