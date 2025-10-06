import os
import sys
import subprocess
from pathlib import Path


def test_analyze_png_is_generated(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"
    data_path = project_root / "tests" / "data" / "COSMIC_v1_SBS_GRCh37.txt"

    output_png = tmp_path / "all_signatures.png"

    cmd = [
        sys.executable,
        str(main_py),
        "--input",
        str(data_path),
        "--output_dir",
        str(tmp_path),
        "--prefix",
        "reprint_",
        "--analyze_png",
        str(output_png),
    ]

    subprocess.run(cmd, check=True)

    assert output_png.exists(), "Expected analyze PNG to be generated"
    assert output_png.stat().st_size > 0, "Generated PNG should not be empty"


