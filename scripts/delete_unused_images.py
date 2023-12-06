from os import environ
from pathlib import Path

root = Path(environ.get("PYTHONPATH", "."))
data = root / "training" / "data"
base2018 = data / "raw" / "2018" / "train_val2018"
base2019 = data / "raw" / "2019" / "train_val2019"
base2021 = data / "raw" / "2021"


def delete_2018_images(path: Path):
    if not path.exists():
        return "Path does not exist"
    for p in path.iterdir():
        if p.is_dir():
            if "Fungi" in p.name:
                continue
            else:
                for d in p.iterdir():
                    for f in d.iterdir():
                        f.unlink()
                    d.rmdir()
                p.rmdir()

def delete_2019_images(path: Path):
    if not path.exists():
        return "Path does not exist"
    for p in path.iterdir():
        if p.is_dir():
            if "Fungi" in p.name:
                continue
            else:
                for d in p.iterdir():
                    for f in d.iterdir():
                        f.unlink()
                    d.rmdir()
                p.rmdir()

def delete_2021_images(path: Path):
    if not path.exists():
        return "Path does not exist"    
    for p in ["train", "val"]:
        for d in (path / p).iterdir():
            if d.is_dir():
                if "Fungi" in d.name:
                    continue
                else:
                    for f in d.iterdir():
                        f.unlink()
                    d.rmdir()


if __name__ == "__main__":
    delete_2018_images(base2018)
    delete_2019_images(base2019)
    delete_2021_images(base2021)
