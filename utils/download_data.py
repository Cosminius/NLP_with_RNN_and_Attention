from pathlib import Path

import requests


def download_text(text_path, url_to_download):
    path = Path(text_path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = url_to_download
        print(url)
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)
    return path.read_text()


shakespeare_text = download_text(
    text_path="datasets/shakespeare/shakespeare.txt",
    url_to_download="https://homl.info/shakespeare",
)
