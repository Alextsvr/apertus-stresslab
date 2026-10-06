"""Package the curated Phase 3B evidence without model weights or other runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from verify_evidence import RAW_NAMES, ROOT, RUN, verify

TOP_FILES = ('.gitignore', '.gitattributes', 'LICENSE', 'README.md', 'pyproject.toml',
             'requirements.txt', 'requirements-inference.txt')
DIRECTORIES = ('src', 'data', 'tests', 'scripts', 'docs', 'notebooks', 'evidence/colab')
SUFFIXES = {'.py', '.md', '.json', '.jsonl', '.txt', '.toml', '.sha256', '.ipynb'}
EXCLUDED = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.cache', 'dist', 'node_modules'}


def payload_files(root: Path) -> list[Path]:
    tracked = None
    if (root / '.git').exists():
        try:
            tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=root,
                          stderr=subprocess.DEVNULL, text=True).splitlines())
        except subprocess.CalledProcessError:
            pass  # A portable/non-Git snapshot uses the existing suffix allowlist.
    selected = [root / path for path in TOP_FILES]
    selected.extend(root / RUN / name for name in RAW_NAMES)
    for directory in DIRECTORIES:
        for path in (root / directory).rglob('*'):
            rel = path.relative_to(root)
            if tracked is not None and rel.as_posix() not in tracked:
                continue
            if (path.is_file() and not path.is_symlink() and not EXCLUDED.intersection(rel.parts)
                    and not any(part.startswith('.') or part.endswith('.egg-info') for part in rel.parts)
                    and (path.suffix in SUFFIXES or path.name == 'SHA256SUMS')):
                selected.append(path)
    catalog_path = root / 'evidence/colab/catalog.json'
    if catalog_path.is_file():
        for item in json.loads(catalog_path.read_text(encoding='utf-8'))['attempts']:
            relative = Path(item['archive'])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[:2] != ('evidence', 'colab'):
                raise ValueError('Unsafe diagnostic archive path')
            selected.extend([root / relative, root / (item['archive'] + '.sha256')])
    for path in selected:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or non-regular package input: {path.relative_to(root)}')
    return sorted(set(selected), key=lambda path: path.relative_to(root).as_posix())


def git_value(root: Path, *args: str) -> str | None:
    if not (root / '.git').exists():
        return None
    try:
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_archive(root: Path, destination: Path) -> tuple[Path, int]:
    root = root.resolve()
    verify(root)
    if (root / 'evidence/colab/catalog.json').is_file():
        from verify_colab_evidence import verify as verify_colab
        verify_colab(root, verify_git=(root / '.git').exists())
    files = payload_files(root)
    payload = {path.relative_to(root).as_posix(): path.read_bytes() for path in files}
    provenance = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': git_value(root, 'rev-parse', 'HEAD'),
        'preregistration_tag_commit': git_value(root, 'rev-parse', 'phase3b-preregistered^{commit}'),
        'git_working_tree_status': git_value(root, 'status', '--short'),
        'curated_run': RUN.as_posix(),
        'note': 'Portable evidence snapshot; no Git history, credentials or model weights. No inference or rescore.',
    }
    payload['PACKAGE_PROVENANCE.json'] = (json.dumps(provenance, indent=2) + '\n').encode('utf-8')
    hashes = ''.join(f'{hashlib.sha256(content).hexdigest()}  {name}\n' for name, content in sorted(payload.items()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(payload.items()):
            archive.writestr(name, content)
        archive.writestr('PACKAGE_SHA256SUMS', hashes.encode('utf-8'))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
        for name, content in payload.items():
            if archive.read(name) != content:
                raise ValueError(f'Archive content differs: {name}')
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + '.sha256').write_text(
        f'{digest}  {destination.name}\n', encoding='utf-8', newline='\n')
    return destination, len(payload) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/apertus-stresslab-phase3b-evidence.zip')
    args = parser.parse_args()
    path, count = build_archive(ROOT, args.output)
    print(f'Created {path} ({count} files); content and SHA256 verified. No inference or rescore.')


if __name__ == '__main__':
    main()
