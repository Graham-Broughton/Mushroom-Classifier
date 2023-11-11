from pathlib import Path
import sys

root = Path(sys.argv[1])
base2018 = root / "2018" / 'train_val2018'
base2021 = root / "2021"

for p in base2018.iterdir():
    if p.is_dir():
        if 'Fungi' in p.name:
            continue
        else:
            for d in p.iterdir():
                for f in d.iterdir():
                    f.unlink()
                d.rmdir()
            p.rmdir()

for p in ['train', 'val']:
    for d in (base2021 / p).iterdir():
        if d.is_dir():
            if 'Fungi' in d.name:
                continue
            else:
                for f in d.iterdir():
                    f.unlink()
                d.rmdir()