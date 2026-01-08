import pandas as pd

def load_viirs(csv_path):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["acq_date"])
    return df[["latitude", "longitude", "date"]]
