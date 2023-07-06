import json
import os
import sys

import numpy as np
import pandas as pd

from simulator import transaction_simulator as ts

os.chdir(sys.path[0])
data_dir = "./ln_data/"
max_threads = 2


def load_temp_data(
    json_files,
    node_keys=["pub_key", "last_update"],
    edge_keys=["node1_pub", "node2_pub", "last_update", "capacity"],
):
    """Load LN graph json files from several snapshots"""
    node_info, edge_info = [], []
    for idx, json_f in enumerate(json_files):
        with open(json_f) as f:
            try:
                tmp_json = json.load(f)
            except json.JSONDecodeError:
                print("JSONDecodeError: " + json_f)
                continue
        new_nodes = pd.DataFrame(tmp_json["nodes"])[node_keys]
        new_edges = pd.DataFrame(tmp_json["edges"])[edge_keys]
        new_nodes["snapshot_id"] = idx
        new_edges["snapshot_id"] = idx
        print(json_f, len(new_nodes), len(new_edges))
        node_info.append(new_nodes)
        edge_info.append(new_edges)
    edges = pd.concat(edge_info)
    edges["capacity"] = edges["capacity"].astype("int64")
    edges["last_update"] = edges["last_update"].astype("int64")
    print("All edges:", len(edges))
    edges_no_loops = edges[edges["node1_pub"] != edges["node2_pub"]]
    print("All edges without loops:", len(edges_no_loops))
    return pd.concat(node_info), edges_no_loops


def generate_directed_graph(
    edges, policy_keys=["disabled", "fee_base_msat", "fee_rate_milli_msat", "min_htlc"]
):
    """Generate directed graph data from undirected payment channels."""
    directed_edges = []
    indices = edges.index
    for idx in tqdm(indices):
        row = edges.loc[idx]
        e1 = [
            row[x]
            for x in [
                "snapshot_id",
                "node1_pub",
                "node2_pub",
                "last_update",
                "channel_id",
                "capacity",
            ]
        ]
        e2 = [
            row[x]
            for x in [
                "snapshot_id",
                "node2_pub",
                "node1_pub",
                "last_update",
                "channel_id",
                "capacity",
            ]
        ]
        if row["node2_policy"] == None:
            e1 += [None for x in policy_keys]
        else:
            e1 += [row["node2_policy"][x] for x in policy_keys]
        if row["node1_policy"] == None:
            e2 += [None for x in policy_keys]
        else:
            e2 += [row["node1_policy"][x] for x in policy_keys]
        directed_edges += [e1, e2]
    cols = [
        "snapshot_id",
        "src",
        "trg",
        "last_update",
        "channel_id",
        "capacity",
    ] + policy_keys
    directed_edges_df = pd.DataFrame(directed_edges, columns=cols)
    return directed_edges_df


def preprocess_json_file(json_file):
    """Generate directed graph data (traffic simulator input format) from json LN snapshot file."""
    json_files = [json_file]
    print("\ni.) Load data")
    EDGE_KEYS = [
        "node1_pub",
        "node2_pub",
        "last_update",
        "capacity",
        "channel_id",
        "node1_policy",
        "node2_policy",
    ]
    nodes, edges = load_temp_data(json_files, edge_keys=EDGE_KEYS)
    print(len(nodes), len(edges))
    print("Remove records with missing node policy")
    print(edges.isnull().sum() / len(edges))
    origi_size = len(edges)
    edges = edges[(~edges["node1_policy"].isnull()) & (~edges["node2_policy"].isnull())]
    print(origi_size - len(edges))
    print("\nii.) Transform undirected graph into directed graph")
    directed_df = generate_directed_graph(edges)
    # print(directed_df.head())
    print("\niii.) Fill missing policy values with most frequent values")
    print("missing values for columns:")
    print(directed_df.isnull().sum())
    directed_df = directed_df.fillna(
        {
            "disabled": False,
            "fee_base_msat": 1000,
            "fee_rate_milli_msat": 1,
            "min_htlc": 1000,
        }
    )
    for col in ["fee_base_msat", "fee_rate_milli_msat", "min_htlc"]:
        directed_df[col] = directed_df[col].astype("float64")
    return directed_df


def diff_edges_single(simulator, percentage=100, amount=0):
    simulator.init_graph()
    G = simulator.G_without_balance.copy()
    edge_sum = len(G.edges())
    capa_lib = []
    rate = 0
    count = 0
    sum_lock = 0
    # get capacity_lib
    for edge in G.edges():
        src, trg = edge
        capa_lib.append(G[src][trg]["capacity"])
    edge_delete = []
    edge_dict = dict()
    for edge in G.edges():
        src, trg = edge
        edge_dict[(src, trg)] = G[src][trg]["capacity"]
    L = sorted(edge_dict.items(), key=lambda item: item[1], reverse=False)
    start = int(percentage * edge_sum / 100)
    end = int((percentage + 10) * edge_sum / 100)
    L = L[start:end]
    for record in L:
        sum_lock += record[1]
        edge_delete.append(record[0])
    lock_rate = sum_lock / sum(capa_lib)
    # print("delete:", len(edge_delete))
    # print("remove:%d" % len(edge_delete))
    G = simulator.G
    for remove_edge in edge_delete:
        src, trg = remove_edge
        if remove_edge in G.edges():
            cap, fee, is_trg, total_cap = simulator.current_capacity_map[(src, trg)]
            G.remove_edge(src, trg)
            if is_trg:
                G.remove_edge(src, trg + "_trg")
    rate = simulator.simulate(
        weight="total_fee",
        with_node_removals=True,
        max_threads=max_threads,
        with_retry=True,
    )
    return rate, lock_rate


def diff_edges_accu(simulator, percentage=100, amount=0):
    simulator.init_graph()
    G = simulator.G_without_balance.copy()
    edge_sum = len(G.edges())
    capa_lib = []
    rate = 0
    count = 0
    sum_lock = 0
    # get capacity_lib
    for edge in G.edges():
        src, trg = edge
        capa_lib.append(G[src][trg]["capacity"])
    threshold = np.percentile(capa_lib, percentage)
    edge_delete = []
    edge_dict = dict()
    for edge in G.edges():
        src, trg = edge
        edge_dict[(src, trg)] = G[src][trg]["capacity"]
    L = sorted(edge_dict.items(), key=lambda item: item[1], reverse=True)
    n = int(edge_sum * (100 - percentage) / 100)
    L = L[:n]
    for record in L:
        sum_lock += record[1]
        edge_delete.append(record[0])
    lock_rate = sum_lock / sum(capa_lib)
    # print("threshold:", threshold)
    # print("remove:%d" % len(edge_delete))
    G = simulator.G
    for remove_edge in edge_delete:
        src, trg = remove_edge
        if remove_edge in G.edges():
            cap, fee, is_trg, total_cap = simulator._map[(src, trg)]
            G.remove_edge(src, trg)
            if is_trg:
                G.remove_edge(src, trg + "_trg")
    print("remove finished, there still are %d edges" % len(G.edges()))
    rate = simulator.simulate(
        weight="total_fee",
        with_node_removals=True,
        max_threads=max_threads,
        with_retry=True,
    )
    return rate, lock_rate


def run_experiment(edges, parameter_file, output_dir, amount):

    print("# 2. Load parameters")
    with open(parameter_file) as f:
        params = json.load(f)
    print(params)

    count = params["count"]
    epsilon = params["epsilon"]
    drop_disabled = params["drop_disabled"]
    drop_low_cap = params["drop_low_cap"]
    with_depletion = params["with_depletion"]
    find_alternative_paths = True

    print("# 3. Load meta data")
    node_meta = pd.read_csv("%s/1ml_meta_data.csv" % data_dir)
    providers = list(node_meta["pub_key"])

    print("# 4. Simulation")
    simulator = ts.TransactionSimulator(
        edges,
        providers,
        amount,
        count,
        drop_disabled=drop_disabled,
        drop_low_cap=drop_low_cap,
        eps=epsilon,
        with_depletion=with_depletion,
    )
    transactions = simulator.transactions
    print("## simulator.init_graph()")
    simulator.init_graph()
    capa_lib = []
    G = simulator.G_without_balance.copy()
    sum_capa = 0
    for edge in G.edges():
        src, trg = edge
        capa_lib.append(G[src][trg]["capacity"])
    threshold = np.percentile(capa_lib, 0)
    balance_total = sum(capa_lib)

    result = dict()
    hub_proportion = 0.015
    # simulator.init_score(threshold, hub_proportion)

    for target in ["size", "length", "amount", "bankrupt", "fee"]:
        if target == "size":
            simulator.init_score_from_file(threshold, hub_proportion)
        elif target == "length":
            simulator.init_score_from_file_length(threshold, hub_proportion)
        elif target == "amount":
            simulator.init_score_from_file_capacity(threshold, hub_proportion)
        elif target == "bankrupt":
            simulator.init_score_from_file_poor_node(threshold, hub_proportion)
        elif target == "fee":
            simulator.init_score_from_file_fee(threshold, hub_proportion)
        (
            cost,
            cost_ratio,
            locked,
            lock_ratio,
            success_rate,
            avg_fee,
            avg_attempt,
            length,
            node_num,
            path_info,
            bankrupt_num,
        ) = simulator.simulate_with_Dos_attack(
            edges, 1000000000000000000000000000000000000000000, 60000
        )
        total_cost = cost
        for i in range(0, 110, 10):
            cost_ratio = i / 100
            print(total_cost * cost_ratio)
            (
                cost,
                cost_ratio_copy,
                locked,
                lock_ratio,
                success_rate,
                fee,
                attempt,
                length,
                node_num,
                path_info,
                bankrupt_num,
            ) = simulator.simulate_with_Dos_attack(
                edges, total_cost * cost_ratio, 60000
            )
            avg_fee = np.mean(list(fee.values()))
            avg_attempt = np.mean(list(attempt.values()))
            result[i] = [
                cost,
                cost_ratio_copy,
                locked,
                lock_ratio,
                success_rate,
                avg_fee,
                avg_attempt,
                node_num,
                bankrupt_num,
            ]
            # get fee and attempt
            result_pd = pd.DataFrame.from_dict(fee, orient="index", columns=["fee"])
            result_pd.to_csv(
                "./analysis/attack_result/"
                + target
                + "_attack_result/fee/"
                + str(amount)
                + "_"
                + str(i)
                + "_result_fee.csv"
            )

            # result_pd = pd.DataFrame.from_dict(path_info, orient="index")
            # result_pd.to_csv("./analysis/attack_result/"+target+"_attack_result/path_info/" +
            #                  str(amount)+"_"+str(i)+"_result_paths.csv")

            # result_pd = pd.DataFrame.from_dict(
            #     attempt, orient="index", columns=["attempt"])
            # result_pd.to_csv("./analysis/attack_result/"+target+"_attack_result/attempt/" +
            #                  str(amount)+"_"+str(i)+"_result_attempt.csv")
            # result_pd = pd.DataFrame.from_dict(
            #     length, orient="index", columns=["length"])
            # result_pd.to_csv("./analysis/attack_result/"+target+"_attack_result/length/" +
            #                  str(amount)+"_"+str(i)+"_result_length.csv")
        result_pd = pd.DataFrame.from_dict(
            result,
            orient="index",
            columns=[
                "cost",
                "cost_ratio",
                "locked",
                "lock_ratio",
                "rate",
                "fee",
                "attempts",
                "node_num",
                "bankrupt_num",
            ],
        )
        result_pd.to_csv(
            "./analysis/attack_result/"
            + target
            + "_attack_result/var_cost/"
            + str(amount)
            + "_result.csv"
        )

    # for hub_proportion in [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
    #     # for cost in :
    #     score = simulator.init_score_from_file(threshold, hub_proportion)
    #     cost = 0
    #     lock = 0
    #     if score == None:
    #         result[hub_proportion] = [0, 0]
    #     else:
    #         for pair in score:
    #             lock += pair[1][0]
    #             cost += pair[1][1]
    #         lock /= balance_total
    #         cost /= balance_total
    #         result[hub_proportion] = [cost, lock]
    # result_pd = pd.DataFrame.from_dict(result, orient="index", columns=[
    #                                    "cost_ratio", "lock_ratio"])
    # result_pd.to_csv("./analysis/result_capacity.csv")


# "args":[ "preprocessed", "0", "./params.json", "../output/"]
if __name__ == "__main__":

    if len(sys.argv) == 5:
        input_type = sys.argv[1]
        print("# 1. Load LN graph data")
        if input_type == "raw":
            json_file = sys.argv[2]
            directed_edges = preprocess_json_file(json_file)
        elif input_type == "preprocessed":
            snapshot_id = int(sys.argv[2])
            snapshots = pd.read_csv("%s/ln_edges.csv" % data_dir)
            directed_edges = snapshots[snapshots["snapshot_id"] == snapshot_id]
        else:
            raise ValueError("The first arguments must be 'raw' or 'preprocessed'!")
        parameter_file = sys.argv[3]
        output_folder = sys.argv[4]
        for amount in [100000]:
            run_experiment(directed_edges, parameter_file, output_folder, amount)

    else:
        print("You must support 4 input arguments:")
        print(
            "   run_simulator.py raw <json_file_path> <parameter_file> <output_folder>"
        )
        print("OR")
        print(
            "   run_simulator.py preprocessed <snapshot_id (int)> <parameter_file> <output_folder>"
        )
