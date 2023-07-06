import csv

import pandas as pd

if __name__ == "__main__":
    snapshot_id = 0
    data_dir = "/Users/ZhiChunLu/research/Congestion/Congestion/Our/ln_data/"
    snapshots = pd.read_csv("%s/ln_edges.csv" % data_dir)
    directed_edges = snapshots[snapshots["snapshot_id"] == snapshot_id]
    directed_edges.to_csv("new.csv")
