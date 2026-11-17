"""Скрипт проверки публикации находит запрещённое содержимое (на временных файлах, без git)."""
import importlib.util

import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("cps", ROOT / "tools" / "check_publish_safety.py")
cps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cps)


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cps, "ROOT", tmp_path)
    return tmp_path


def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return rel


@pytest.mark.parametrize("rel,text,expect", [
    ("model.txt", "tree\n", ".txt вне requirements"),
    ("data.csv", "a,b\n1,2\n", "CSV вне шаблона"),
    ("x.parquet", "x", "запрещённый тип"),
    # примеры собираются из частей, чтобы в исходном тексте теста не было «настоящих» совпадений
    ("a.py", "P = r'" + "C:" + "\\" + "Users" + "\\" + "someone'\n", "абсолютный путь Windows"),
    ("b.py", "P = '/" + "home/someone/file'\n", "абсолютный путь Unix"),
    ("c.md", "token hf" + "_" + "A" * 30 + "\n", "токен Hugging Face"),
    ("d.toml", "pass" + 'word = "secret"\n', "пароль"),
    ("e.md", "contact: someone" + "@" + "example.com\n", "электронной почты"),
    ("templates/features_template.csv", "a,b\n1,2\n", "только строка заголовков"),
])
def test_detects(tmp_root, rel, text, expect):
    write(tmp_root, rel, text)
    assert any(expect in p for p in cps.scan([rel]))


def test_clean_files_pass(tmp_root):
    files = [write(tmp_root, "ok.py", "x = 1\n"), write(tmp_root, "requirements.txt", "pandas==3.1.1\n"),
             write(tmp_root, "templates/features_template.csv", "a,b\n")]
    assert cps.scan(files) == []
