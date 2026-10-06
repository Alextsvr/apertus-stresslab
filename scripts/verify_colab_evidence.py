"""Verify published Colab evidence using stdlib only; never infer or rescore."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path("evidence/colab/catalog.json")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and ".." not in p.parts and "\\" not in name
            and ":" not in name and p.as_posix() == name, f"Unsafe evidence path: {name}")
    return name


def check_manifest(archive, expected_count):
    names = archive.namelist()
    require(len(names) == len(set(names)), "Duplicate ZIP entry")
    for name in names:
        safe_name(name)
    entries = []
    for line in archive.read("SHA256SUMS").decode("utf-8").splitlines():
        h, name = line.split("  ", 1)
        safe_name(name)
        entries.append((h, name))
    require(len(entries) == expected_count and len({n for _, n in entries}) == len(entries)
            and {n for _, n in entries} == set(names) - {"SHA256SUMS"}, "Incomplete or duplicate internal manifest")
    for h, name in entries:
        require(digest(archive.read(name)) == h, f"Internal SHA256 mismatch: {name}")
    return len(entries)


def verify(root=ROOT, *, verify_git=True, include_phase3b=True):
    root = Path(root)
    catalog = json.loads((root / CATALOG).read_text(encoding="utf-8"))
    require(catalog["schema_version"] == 1, "Unknown catalog schema")
    items = catalog["attempts"]
    require(len({item["id"] for item in items}) == len(items), "Duplicate attempt ID")
    checked = []
    for item in items:
        path = root / safe_name(item["archive"])
        content = path.read_bytes()
        require(len(content) == item["archive_bytes"] and digest(content) == item["archive_sha256"],
                f"Archive SHA256/size mismatch: {item['id']}")
        checksum = path.with_suffix(".zip.sha256").read_text(encoding="utf-8").split()
        require(checksum == [item["archive_sha256"], path.name], f"Archive checksum mismatch: {item['id']}")
        for field in ("report", "audit_metadata", *(["annotations"] if "annotations" in item else [])):
            rel = safe_name(item[field])
            require(digest((root / rel).read_bytes()) == item[field + "_sha256"], f"Changed {field}: {item['id']}")
        with zipfile.ZipFile(path) as archive:
            internal_count = check_manifest(archive, item["internal_files"])
            for source, member in item["registered_sources"].items():
                safe_name(source)
                require(member in archive.namelist(), f"Missing registered source: {member}")
                if verify_git:
                    expected = subprocess.check_output(["git", "show", f"{item['procedure_commit']}:{source}"], cwd=root)
                    require(archive.read(member) == expected, f"Source differs from procedure commit: {source}")
            result = json.loads(archive.read(item["result_member"]))
            for key, expected in item["stored_result"].items():
                require(result.get(key) == expected, f"Stored result differs: {item['id']} / {key}")
            if "preflight.json" in archive.namelist():
                preflight = json.loads(archive.read("preflight.json"))
                require(preflight["git_commit"] == item["procedure_commit"], f"Preflight commit differs: {item['id']}")
                if verify_git:
                    for rel, h in preflight["source_sha256"].items():
                        safe_name(rel)
                        expected = subprocess.check_output(["git", "show", f"{item['procedure_commit']}:{rel}"], cwd=root)
                        require(digest(expected) == h, f"Frozen source differs: {rel}")
            counts = {}
            for mode, spec in item.get("stored_record_counts", {}).items():
                records = [json.loads(line) for line in archive.read(spec["member"]).splitlines()]
                actual = dict(Counter(record["status"] for record in records))
                require(len(records) == spec["records"] and actual == spec["statuses"], f"Stored counts differ: {mode}")
                counts[mode] = actual
            checked.append({"id": item["id"], "archive_sha256": "match", "internal_hashes": f"{internal_count}/{internal_count}",
                            "registered_sources_checked_against_git": verify_git, "stored_counts": counts})
    manifest = root / "evidence/colab/SHA256SUMS"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        h, rel = line.split("  ", 1)
        require(digest((root / safe_name(rel)).read_bytes()) == h, f"Publication SHA256 mismatch: {rel}")
    result = {"attempts_verified": len(checked), "attempts": checked,
              "inference_or_rescoring_performed": False}
    if include_phase3b:
        from verify_evidence import verify as verify_phase3b
        result["original_phase3b"] = verify_phase3b(root)
    if (root / 'evidence/colab/semantic/catalog.json').is_file():
        from verify_semantic_evidence import verify as verify_semantic
        result['new_semantic_study'] = verify_semantic(root, verify_git=verify_git)
    if (root / 'evidence/colab/int8_matched/catalog.json').is_file():
        from verify_int8_matched_evidence import verify as verify_matched
        result['int8_matched_condition'] = verify_matched(root, verify_git=verify_git)
    if (root / 'evidence/colab/date_ablation/catalog.json').is_file():
        from verify_date_ablation_evidence import verify as verify_dates
        result['date_context_ablation'] = verify_dates(root, verify_git=verify_git)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
