import json
from types import SimpleNamespace

import pytest

from . import generate as generation


def spec():
    return {
        "files": {"calc.py": "VALUE = 0\n"},
        "reference_files": {"calc.py": "VALUE = 1\n"},
        "checks": [{"type": "file_contains", "path": "calc.py", "text": "VALUE = 1"}],
        "expected_tools": ["replace_in_file"], "criteria": "把常量改为 1",
    }


def test_generate_and_import(monkeypatch, tmp_path):
    monkeypatch.setattr(generation, "PROJECT_ROOT", tmp_path)
    (tmp_path / "evals" / "datasets").mkdir(parents=True)
    monkeypatch.setattr(generation, "load_cases", lambda: [])
    monkeypatch.setattr(generation, "get_judge", lambda: SimpleNamespace(get_model_name=lambda: "mock"))
    monkeypatch.setattr(generation, "FiltrationConfig", lambda **kwargs: kwargs)
    captured = {}

    class FakeSynthesizer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate_goldens_from_contexts(self, **kwargs):
            assert kwargs["_send_data"] is False
            return [SimpleNamespace(input="修复常量，不修改测试", expected_output=json.dumps(spec()))]

    monkeypatch.setattr(generation, "Synthesizer", FakeSynthesizer)
    path = tmp_path / "draft.json"
    generation.generate(SimpleNamespace(context=[], count=1, output=path))
    drafts = json.loads(path.read_text())
    assert not drafts[0]["reviewed"]
    assert captured["styling_config"].task
    with pytest.raises(ValueError, match="reviewed"):
        generation.import_reviewed(path)
    drafts[0]["reviewed"] = True
    path.write_text(json.dumps(drafts))
    output = generation.import_reviewed(path)
    imported = json.loads(output.read_text())
    assert "reference_files" not in imported[0]
    assert imported[0]["files"] == spec()["files"]
    with pytest.raises(ValueError, match="已存在"):
        generation.import_reviewed(path)


def test_reject_unsafe_paths():
    invalid = spec()
    invalid["files"] = {"../escape.py": "pass"}
    with pytest.raises(ValueError, match="路径"):
        generation.validate_spec(invalid)
