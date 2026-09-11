#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from shutil import copy2

in_csv="../../data/synth_metadata_filtered2.csv"
out_dir="../../data/imagefolder/synth"

#columnas
path_column="path"
class_column="superclass"
split_column="split"

Path(out_dir).mkdir(parents=True, exist_ok=True)

# Read metadata
df=pd.read_csv(in_csv)

# Create directories and copy
for i,row in df.iterrows():
    src=Path(str(row[path_column])).resolve()
    split=str(row[split_column])
    cls=str(row[class_column])

    dst_dir=Path(out_dir)/split/cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst=dst_dir/(src.name)
    try:
        copy2(src, dst)
    except Exception:
        pass
