from fastbook import *
from kaggle import api
from pathlib import Path

path = Path('./data')
if not path.exists():
    path.mkdir()
    api.competition_download_cli('bluebook-for-bulldozers', path=path)
    file_extract(path/'bluebook-for-bulldozers.zip')
