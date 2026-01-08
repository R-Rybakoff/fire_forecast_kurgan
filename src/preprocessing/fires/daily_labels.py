def daily_fire_labels(df):
    daily = (
        df.groupby(["cell_id", "date"])
        .size()
        .reset_index(name="fire")
    )
    daily["fire"] = 1
    return daily
