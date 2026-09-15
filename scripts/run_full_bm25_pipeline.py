"""
run_full_bm25_pipeline.py

Wait for WikiExtractor to finish, then run the full BM25 pipeline:
  build_corpus_from_dump --resume -> prepare_pyserini_collection -> index -> eval

Usage:
    py scripts/run_full_bm25_pipeline.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JDK = ROOT / "data/tools/jdk/jdk-21.0.6+7"


def setup_java():
    if JDK.joinpath("bin/java.exe").exists():
        os.environ["JAVA_HOME"] = str(JDK)
        os.environ["PATH"] = str(JDK / "bin") + os.pathsep + os.environ.get("PATH", "")


def wikiextractor_running() -> bool:
    """Return True while any Python process is running WikiExtractor."""
    ps_cmd = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "ForEach-Object { $_.CommandLine } | "
        "Where-Object { $_ -match 'wikiextractor' } | "
        "Measure-Object | Select-Object -ExpandProperty Count"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore",
        ).strip()
        return out not in ("", "0")
    except Exception:
        return False


def wait_for_wikiextractor():
    print("Waiting for WikiExtractor to finish ...", flush=True)
    stable_rounds = 0
    last_count = -1
    while True:
        if wikiextractor_running():
            stable_rounds = 0
            print(f"  WikiExtractor still running ...", flush=True)
            time.sleep(300)
            continue

        count = sum(1 for _ in (ROOT / "data/arwiki/extracted").rglob("wiki_*") if _.is_file())
        if count == last_count:
            stable_rounds += 1
        else:
            stable_rounds = 0
            last_count = count
        if stable_rounds >= 2:
            break
        print(f"  extract files={count}, waiting for stability ...", flush=True)
        time.sleep(300)


def run(cmd: list[str]):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    setup_java()

    wait_for_wikiextractor()
    print("WikiExtractor done.", flush=True)

    run([sys.executable, "scripts/build_corpus_from_dump.py", "--max-window", "1", "--resume"])

    collection_dir = ROOT / "data/arafa/pyserini_collection"
    if collection_dir.exists():
        import shutil
        shutil.rmtree(collection_dir)
    run([sys.executable, "scripts/prepare_pyserini_collection.py"])

    index_dir = ROOT / "data/arafa/bm25_index"
    if index_dir.exists():
        import shutil
        shutil.rmtree(index_dir)
    run([
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", "data/arafa/pyserini_collection",
        "--index", "data/arafa/bm25_index",
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--language", "ar",
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ])

    run([sys.executable, "scripts/run_bm25_eval.py"])
    print("\nFull pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
