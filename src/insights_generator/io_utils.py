from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi import UploadFile


def load_dataframe_from_upload(file: UploadFile) -> pd.DataFrame:
    filename = (file.filename or "").lower()
    raw = file.file.read()
    if filename.endswith(".csv"):
        return pd.read_csv(BytesIO(raw))
    if filename.endswith(".parquet"):
        return pd.read_parquet(BytesIO(raw))
    raise ValueError("Unsupported file type. Upload .csv or .parquet.")
