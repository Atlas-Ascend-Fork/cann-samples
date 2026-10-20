# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Use real executable version-query fixtures with the complete checker."""
import importlib.util
import logging
from pathlib import Path
import sys

import pytest


def load_checker():
    source = Path(__file__).resolve().parents[2] / "scripts/check_env.py"
    spec = importlib.util.spec_from_file_location("sample_failed_version_query", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command_file(tmp_path, text, status, stream):
    path = tmp_path / "version-query-tool"
    path.write_text(f"#!{sys.executable}\nimport sys\nsys.{stream}.write({text!r})\nsys.exit({status})\n")
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize("text", ["error references tool 99.0", "error references tool 1.0"])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_failed_queries_are_warnings_not_version_comparisons(tmp_path, capsys, monkeypatch, text, stream):
    checker = load_checker()
    logger = logging.getLogger("sample_failed_query_test")
    logger.handlers = [logging.StreamHandler(sys.stdout)]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    monkeypatch.setattr(checker, "_logger", logger)
    tool = command_file(tmp_path, text, 2, stream)
    assert checker.check_command(tool, min_version="3.10") is True
    output = capsys.readouterr().out
    assert "[WARN]" in output
    assert "[PASS]" not in output and "[FAIL]" not in output
    assert "status 2" in output


@pytest.mark.parametrize("text, expected", [("tool 99.0", True), ("tool 1.0", False)])
def test_successful_query_versions_keep_existing_comparison(tmp_path, text, expected):
    checker = load_checker()
    tool = command_file(tmp_path, text, 0, "stdout")
    assert checker.check_command(tool, min_version="3.10") is expected


def test_failed_unparseable_query_keeps_warning_fallback(tmp_path):
    checker = load_checker()
    tool = command_file(tmp_path, "query not available", 2, "stderr")
    assert checker.check_command(tool, min_version="3.10") is True
