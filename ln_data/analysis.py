import csv
import os
import numpy as np
if __name__ == "__main__":
    homedir = os.getcwd()
    path_lib = dict()
    hub_proportion = 0.015
    with open(homedir+"/Our/ln_data/paths_"+str(hub_proportion)+".csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            key = row[0][2:-2]
            key = key.replace("'", "")
            src, trg = key.split(", ")
            key = (src, trg)
            if key in path_lib.keys():
                break
            path_lib[key] = []
            for pos in range(1, len(row)):
                path = row[pos][1:-2]
                path = path.replace("'", "")
                path = path.split(", ")
                if path == [""]:
                    continue
                path_lib[key].append(path)
    length = dict()
    for key in path_lib.keys():
        length[key] = round(np.mean([len(i) for i in path_lib[key]]), 2)
    print(np.std(list(length.values())))
    print(np.mean(list(length.values()))-1)
    print(list(length.values()))
