"""Download Karpathy's stories15M checkpoint and the Llama 2 tokenizer into models/.

    python3 tools/download_stories15m.py

Files are skipped if they already exist. models/ is ignored by git.
"""

import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = {
    "stories15M.bin": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
    "tokenizer.bin": "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin",
}


def main() -> None:
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dest = models / name
        if dest.exists():
            print(f"{dest} already exists")
            continue
        print(f"downloading {url}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        print(f"  {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
