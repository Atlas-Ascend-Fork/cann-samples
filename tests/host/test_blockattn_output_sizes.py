# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Use the full verifier CLI with actual temporary binary files, without NPU."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "Samples/2_Performance/block_attn_res_story/block_attn_res_recipes/scripts/verify_result.py"
OUTPUTS = {
    "prepare_numerator": (8, np.float32),
    "prepare_logit_max": (4, np.float32),
    "prepare_exp_sum": (4, np.float32),
    "update_partial": (4, np.float32),
    "update_h": (4, np.uint16),
}


def write_outputs(root):
    output = root / "output"
    output.mkdir()
    for name, (count, dtype) in OUTPUTS.items():
        values = np.ones(count, dtype=dtype)
        for prefix in ("npu", "golden"):
            values.tofile(output / f"{prefix}_{name}.bin")


def run_cli(root, mode="e2e", t=2, s=2, d=2):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--mode", mode, "--t", str(t), "--s", str(s), "--d", str(d)],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("name", list(OUTPUTS))
@pytest.mark.parametrize("damage", ["empty", "short", "long", "partial_element"])
def test_matching_corrupt_files_cannot_pass_declared_shape(tmp_path, name, damage):
    write_outputs(tmp_path)
    for prefix in ("npu", "golden"):
        path = tmp_path / "output" / f"{prefix}_{name}.bin"
        original = path.read_bytes()
        _, dtype = OUTPUTS[name]
        element_size = np.dtype(dtype).itemsize
        if damage == "empty":
            changed = b""
        elif damage == "short":
            changed = original[:-element_size]
        elif damage == "long":
            changed = original + original[:element_size]
        else:
            changed = original + b"x"
        path.write_bytes(changed)
    before = {path.name: path.read_bytes() for path in (tmp_path / "output").iterdir()}
    result = run_cli(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "file size" in result.stdout
    assert "block_attn_res_e2e: FAILED" in result.stdout
    assert before == {path.name: path.read_bytes() for path in (tmp_path / "output").iterdir()}


@pytest.mark.parametrize("mode", ["prepare", "update", "e2e"])
def test_matching_full_outputs_keep_success(tmp_path, mode):
    write_outputs(tmp_path)
    if mode == "prepare":
        ignored = "update"
    elif mode == "update":
        ignored = "prepare"
    else:
        ignored = None
    for path in (tmp_path / "output").iterdir():
        if ignored is not None and ignored in path.name:
            path.unlink()
    result = run_cli(tmp_path, mode)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"block_attn_res_{mode}: PASSED" in result.stdout


@pytest.mark.parametrize("name", ["prepare_numerator", "update_partial"])
def test_full_sized_numerical_mismatch_still_fails(tmp_path, name):
    write_outputs(tmp_path)
    count, dtype = OUTPUTS[name]
    np.full(count, 100, dtype=dtype).tofile(tmp_path / "output" / f"npu_{name}.bin")
    result = run_cli(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "standard=isclose" in result.stdout


@pytest.mark.parametrize("dimensions", [{"t": 0}, {"s": -1}, {"d": 0}, {"d": 8193}])
def test_dimensions_use_generator_valid_range(tmp_path, dimensions):
    write_outputs(tmp_path)
    result = run_cli(tmp_path, **dimensions)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "t/s must be positive and d must be in [1, 8192]" in result.stderr
