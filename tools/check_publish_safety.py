"""Проверка перед коммитом / публикацией: что именно попадёт в git-репозиторий и нет ли там лишнего.

    python tools/check_publish_safety.py            # через git: git ls-files --cached --others --exclude-standard
    python tools/check_publish_safety.py --no-git   # без git: упрощённый разбор .gitignore (для предварительной проверки)

Код возврата: 0 — проблем нет, 1 — найдены проблемы, 2 — git недоступен или папка не является репозиторием.
Скрипт ничего не изменяет, не создаёт коммитов и ничего не отправляет в сеть.
"""
import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 1_000_000
ALLOWED_CSV = {"templates/features_template.csv"}
FORBIDDEN_EXT = {".parquet", ".feather", ".arrow", ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".bin", ".onnx",
                 ".h5", ".hdf5", ".pt", ".pth", ".ckpt", ".safetensors", ".cbm", ".model", ".lgb", ".lgbm",
                 ".zip", ".gz", ".tar", ".7z", ".rar", ".xls", ".xlsx", ".tsv", ".log", ".key", ".pem", ".p12", ".pfx"}
ALLOWED_TXT = {"requirements.txt", "requirements-dev.txt"}
TEXT_EXT = {".py", ".md", ".toml", ".json", ".txt", ".csv", ".cfg", ".ini", ".yml", ".yaml", "", ".example"}
PATTERNS = {
    "абсолютный путь Windows": re.compile(r"\b[A-Za-z]:\\(?!\\)"),
    "абсолютный путь Unix (домашняя папка)": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+"),
    "похоже на токен Hugging Face": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "приватный ключ": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "пароль в явном виде": re.compile(r"(?i)\bpassword\s*[:=]\s*['\"][^'\"]+['\"]"),
    "адрес электронной почты": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}
SENTINELS = {  # путь -> должен ли git его игнорировать
    "lgbm_B_full.txt": True, "any_model_name.txt": True, "models/x.json": True, "data/x.csv": True,
    "validation_2000_features.csv": True, ".streamlit/secrets.toml": True, "nested/secrets.toml": True, ".env": True,
    "x.parquet": True, "x.pkl": True, "run.log": True,
    "requirements.txt": False, "requirements-dev.txt": False, "templates/features_template.csv": False,
    ".streamlit/secrets.toml.example": False, "README.md": False, "LICENSE": False,
}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


def files_via_git():
    if not shutil.which("git"):
        print("git не найден в PATH"); sys.exit(2)
    if git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        print("папка не является git-репозиторием (сначала: git init)"); sys.exit(2)
    out = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").stdout
    files = sorted(f for f in out.split("\0") if f)
    wrong = {}
    for p, should in SENTINELS.items():
        ignored = git("check-ignore", "-q", "--no-index", p).returncode == 0
        if ignored != should:
            wrong[p] = "игнорируется" if ignored else "НЕ игнорируется"
    return files, wrong


def files_without_git():
    rules = []
    for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            rules.append((s.startswith("!"), s.lstrip("!")))

    def ignored(rel):
        parts, res = rel.split("/"), False
        for neg, pat in rules:
            if pat.endswith("/"):
                hit = any(fnmatch.fnmatch(p, pat[:-1]) for p in parts[:-1])
            elif "/" in pat:
                hit = fnmatch.fnmatch(rel, pat)
            else:
                hit = any(fnmatch.fnmatch(p, pat) for p in parts)
            if hit:
                res = not neg
        return res

    files = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                   if p.is_file() and ".git" not in p.relative_to(ROOT).parts and not ignored(p.relative_to(ROOT).as_posix()))
    wrong = {p: ("игнорируется" if ignored(p) else "НЕ игнорируется") for p, s in SENTINELS.items() if ignored(p) != s}
    return files, wrong


def scan(files):
    problems = []
    for rel in files:
        p = ROOT / rel
        ext = p.suffix.lower()
        if ext in FORBIDDEN_EXT:
            problems.append(f"{rel}: запрещённый тип файла {ext}")
        if ext == ".csv" and rel not in ALLOWED_CSV:
            problems.append(f"{rel}: CSV вне шаблона (возможно, данные)")
        if ext == ".txt" and p.name not in ALLOWED_TXT:
            problems.append(f"{rel}: .txt вне requirements (возможно, модель LightGBM)")
        if p.stat().st_size > MAX_BYTES:
            problems.append(f"{rel}: размер {p.stat().st_size:,} байт > {MAX_BYTES:,}")
        if ext in TEXT_EXT or p.name in {"LICENSE", ".gitignore", ".gitattributes"}:
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                problems.append(f"{rel}: не UTF-8 текст"); continue
            for name, rx in PATTERNS.items():
                for m in rx.finditer(text):
                    line = text.count("\n", 0, m.start()) + 1
                    problems.append(f"{rel}:{line}: {name}: «{m.group(0)[:40]}»")
        if rel == "templates/features_template.csv":
            lines = p.read_text(encoding="utf-8").splitlines()
            if len(lines) != 1:
                problems.append(f"{rel}: в шаблоне {len(lines)} строк — должна быть только строка заголовков")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-git", action="store_true", help="упрощённая проверка без git")
    a = ap.parse_args()
    files, wrong = files_without_git() if a.no_git else files_via_git()
    print(f"режим: {'без git (упрощённый)' if a.no_git else 'git'}; файлов к публикации: {len(files)}")
    for f in files:
        print("  ", f)
    problems = scan(files) + [f"правило .gitignore: {p} {s}" for p, s in wrong.items()]
    if problems:
        print(f"\nНАЙДЕНО ПРОБЛЕМ: {len(problems)}")
        for x in problems:
            print("  -", x)
        sys.exit(1)
    print("\nпроблем не найдено")


if __name__ == "__main__":
    main()
