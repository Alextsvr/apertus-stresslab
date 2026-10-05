"""Read-only verification of published evidence; no evaluator or model imports."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('results/2026-10-02_111432')
AUDIT = Path('docs/audits/phase3b-2026-10-02_111432')
CHECKPOINT = 'ed64ba8cb382073d652b24e647a898c94f8ce186'
RAW_NAMES = ('failures.jsonl', 'metadata.json', 'results.jsonl', 'summary.json')


class EvidenceError(ValueError):
    """The evidence is incomplete or differs from its captured state."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def verify(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifest = read_json(root / AUDIT / 'integrity.json')
    expected_paths = {(RUN / name).as_posix() for name in RAW_NAMES}
    entries = manifest['files']
    require(len(entries) == 4 and {entry['path'] for entry in entries} == expected_paths,
            'Integrity record must name exactly the four curated raw files')
    require(manifest['preregistration_commit'] == CHECKPOINT, 'Unexpected preregistration checkpoint')
    for entry in entries:
        payload = (root / entry['path']).read_bytes()
        require(len(payload) == entry['bytes'], f"Size mismatch: {entry['path']}")
        require(hashlib.sha256(payload).hexdigest() == entry['sha256'], f"SHA256 mismatch: {entry['path']}")
    sums = '\n'.join(f"{entry['sha256']}  {entry['path']}" for entry in entries) + '\n'
    require((root / AUDIT / 'SHA256SUMS').read_text(encoding='utf-8') == sums,
            'SHA256SUMS differs from integrity.json')

    records = read_jsonl(root / RUN / 'results.jsonl')
    summary = read_json(root / RUN / 'summary.json')
    metadata = read_json(root / RUN / 'metadata.json')
    cases = read_jsonl(root / 'data/test_cases/false_premise_heldout.jsonl')
    annotations = read_json(root / AUDIT / 'semantic_annotations.json')['records']
    case_map = {case['id']: case for case in cases}
    ids = [r['test_id'] for r in records]
    require(len(records) == len(cases) == len(case_map) == len(set(ids)) == 30,
            'Expected 30 unique records and 30 unique cases')
    require(set(ids) == set(case_map), 'Raw records and dataset IDs differ')
    require(metadata['environment']['git_commit'] == CHECKPOINT, 'Run metadata checkpoint differs')
    for record in records:
        case = case_map[record['test_id']]
        prompt = f"{case['instruction']}\n\nContext:\n{case['context']}\n\nQuestion:\n{case['question']}"
        require(record['base_prompt'] == prompt, f"Prompt mismatch: {record['test_id']}")
        require(record['expected_behavior'] == case['expected'], f"Expectation mismatch: {record['test_id']}")
        require(record['error'] is None and not record['usage']['hit_max_new_tokens'],
                f"Unexpected error/truncation: {record['test_id']}")
    status = dict(Counter(r['status'] for r in records))
    require(status == summary['by_status'] == {'PASS': 22, 'POTENTIAL_FAILURE': 6, 'DETECTED_FAILURE': 2},
            'Stored whole-run status counts are inconsistent')
    fp = [r for r in records if not r['heldout']['control']]
    controls = [r for r in records if r['heldout']['control']]
    require(len(fp) == 24 and len(controls) == 6, 'False-premise/control counts differ')
    require(Counter(r['status'] for r in fp) == {'PASS': 16, 'POTENTIAL_FAILURE': 6, 'DETECTED_FAILURE': 2},
            'Stored false-premise counts differ')
    require(all(r['status'] == 'PASS' for r in controls), 'Stored control outcomes differ')
    require(Counter(r['checks']['false_premise_check']['outcome'] for r in fp) ==
            {'corrected': 16, 'accepted': 2, 'ambiguous': 2, 'not_confirmed': 4},
            'Stored premise-outcome counts differ')
    require(read_jsonl(root / RUN / 'failures.jsonl') == [r for r in records if r['status'] != 'PASS'],
            'Failures file is not the exact stored non-PASS subset')
    require(len(annotations) == 30 and {a['test_id'] for a in annotations} == set(ids),
            'Semantic annotations do not cover all 30 unique IDs')
    raw_by_id = {r['test_id']: r for r in records}
    for annotation in annotations:
        record = raw_by_id[annotation['test_id']]
        require(annotation['stored_response'] == record['response'] and
                annotation['stored_base_prompt'] == record['base_prompt'] and
                annotation['frozen_automated']['status'] == record['status'],
                f"Annotation source mismatch: {annotation['test_id']}")
    return {'raw_sha256': '4/4 match', 'records': 30, 'false_premise': 24, 'controls': 6,
            'stored_status_counts': status, 'annotation_source_matches': 30,
            'inference_or_rescoring_performed': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Extracted package or repository root')
    args = parser.parse_args()
    try:
        result = verify(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Evidence verification failed: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
