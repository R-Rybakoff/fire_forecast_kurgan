import numpy as np

def label_ndvi(ndvi, fires_daily):
    fire_index = set(
        zip(
            fires_daily.cell_id.values,
            fires_daily.date.values
        )
    )

    ndvi["y"] = [
        1 if (cid, d) in fire_index else 0
        for cid, d in zip(ndvi.cell_id, ndvi.date)
    ]

    ndvi["y"] = ndvi["y"].astype("int8")
    return ndvi
