import json
import sys
from types import SimpleNamespace

import pytest

from my_agent.tools.base import ToolContext
from my_agent.tools.shell import execute_command
from .metrics import _check


def result(tmp_path):
    return SimpleNamespace(
        before={"app.py": "old", "tests/test_app.py": "original"},
        after={"app.py": "new", "tests/test_app.py": "original"},
        record=SimpleNamespace(final_output="", tool_calls=[]), workspace=tmp_path,
    )


@pytest.mark.parametrize("selector", [{"path": "tests/test_app.py"}, {"paths": ["tests/test_app.py"]}])
def test_protect_test_not_application(tmp_path, selector):
    r = result(tmp_path)
    check = {"type": "unchanged", **selector}
    assert _check(r, check)[0]
    r.after["tests/test_app.py"] = "tampered"
    assert not _check(r, check)[0]
    del r.after["tests/test_app.py"]
    assert not _check(r, check)[0]


def test_unchanged_default_and_invalid(tmp_path):
    r = result(tmp_path)
    assert not _check(r, {"type": "unchanged"})[0]
    with pytest.raises(ValueError):
        _check(r, {"type": "unchanged", "paths": "tests/test_app.py"})


def test_python_environment(tmp_path, monkeypatch):
    assert sys.version_info[:2] == (3, 12)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    ctx = ToolContext(workspace=tmp_path, ui=None)
    for executable in ("python", "python3"):
        command = f'{executable} -c "import sys; assert sys.version_info[:2] == (3, 12); print(sys.prefix)"'
        response = json.loads(execute_command(ctx, command, requires_approval=False))
        assert response["exitCode"] == 0, response
        assert response["stdout"].strip() == sys.prefix
        assert _check(result(tmp_path), {"type": "command", "cmd": command})[0]
    assert _check(result(tmp_path), {"type": "command", "cmd":
        '{python} -c "import sys; assert sys.version_info[:2] == (3, 12)"'})[0]
