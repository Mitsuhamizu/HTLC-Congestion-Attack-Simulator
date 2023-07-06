import collections
import concurrent.futures
import copy
import csv
import datetime
import functools
import json
import os
import random
import sys
from collections import Counter
from itertools import combinations
from math import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bloom_filter import BloomFilter
from networkx.algorithms.flow import shortest_augmenting_path
from scipy.stats import gaussian_kde
from tqdm import tqdm

from .graph_preprocessing import *
from .path_searching import get_shortest_paths, process_path
from .transaction_sampling import sample_transactions


def shortest_paths_with_exclusion(
    capacity_map, G, cost_prefix, weight, hash_bucket_item
):
    node, bucket_transactions = hash_bucket_item
    H = G.copy()
    H.remove_node(node)
    if node + "_trg" in G.nodes():
        H.remove_node(node + "_trg")  # delete node copy as well
    new_paths, _, _, _ = get_shortest_paths(
        capacity_map,
        H,
        bucket_transactions,
        hash_transactions=False,
        cost_prefix=cost_prefix,
        weight=weight,
    )
    new_paths["node"] = node
    return new_paths


def get_shortest_paths_with_node_removals(
    capacity_map, G, hashed_transactions, cost_prefix="", weight=None, threads=4
):
    print("Parallel execution on %i threads in progress.." % threads)
    if threads > 1:
        f_partial = functools.partial(
            shortest_paths_with_exclusion, capacity_map, G, cost_prefix, weight
        )
        executor = concurrent.futures.ProcessPoolExecutor(threads)
        alternative_paths = list(executor.map(f_partial, hashed_transactions.items()))
        executor.shutdown()
    else:
        alternative_paths = []
        for hash_bucket_item in tqdm(hashed_transactions.items(), mininterval=10):
            alternative_paths.append(
                shortest_paths_with_exclusion(
                    capacity_map, G, cost_prefix, weight, hash_bucket_item
                )
            )
    return pd.concat(alternative_paths)


class TransactionSimulator:
    def __init__(
        self,
        edges,
        merchants,
        amount_sat,
        k,
        eps=0.8,
        drop_disabled=True,
        drop_low_cap=True,
        with_depletion=True,
        time_window=None,
        verbose=False,
    ):
        self.verbose = verbose
        self.with_depletion = with_depletion
        self.amount = amount_sat
        self.edges = prepare_edges_for_simulation(
            edges,
            amount_sat,
            drop_disabled,
            drop_low_cap,
            time_window,
            verbose=self.verbose,
        )
        self.edges_all = prepare_edges_for_simulation(
            edges, amount_sat, drop_disabled, False, time_window, verbose=self.verbose
        )
        self.node_variables, self.merchants, active_ratio = init_node_params(
            self.edges, merchants, verbose=self.verbose
        )
        self.transactions = sample_transactions(
            self.node_variables,
            amount_sat,
            k,
            eps,
            self.merchants,
            verbose=self.verbose,
        )
        self.params = {
            "amount": amount_sat,
            "count": k,
            "epsilon": eps,
            "with_depletion": with_depletion,
            "drop_disabled": drop_disabled,
            "drop_low_cap": drop_low_cap,
            "time_window": time_window,
        }

    def init_graph(self):
        if self.with_depletion:
            # current_capacity_map: 1. balance 2. total_fee 3.whether is trg or not 4.capacity
            # edges_with capacity: 1. src 2. trg 3. capacity 4.fee
            self.current_capacity_map, self.edges_with_capacity = init_capacities(
                self.edges, self.transactions, self.amount, self.edges_all, self.verbose
            )
            self.G = generate_graph_for_path_search(
                self.edges_with_capacity, self.transactions, self.amount
            )
            self.G_without_balance = nx.from_pandas_edgelist(
                self.edges_all,
                source="src",
                target="trg",
                edge_attr=["capacity", "total_fee"],
                create_using=nx.DiGraph(),
            )
        else:
            self.current_capacity_map = None
            self.G = generate_graph_for_path_search(
                self.edges, self.transactions, self.amount
            )

    def simulate(
        self,
        weight=None,
        with_node_removals=True,
        max_threads=2,
        excluded=[],
        required_length=None,
        with_retry=False,
    ):
        G = self.G
        current_capacity_map = self.current_capacity_map
        # 1. balance 2. total_fee 3.whether is trg or not 4.capacity
        # it empty at the beginning
        if len(excluded) > 0:
            print(G.number_of_edges(), G.number_of_nodes())
            for node in excluded:
                if node in G.nodes():
                    G.remove_node(node)
                pseudo_node = str(node) + "_trg"
                if pseudo_node in G.nodes():
                    G.remove_node(pseudo_node)
            if self.verbose:
                print(G.number_of_edges(), G.number_of_nodes())
            print("Additional nodes were EXCLUDED!")
        print("Graph and capacities were INITIALIZED")
        if self.verbose:
            print("Using weight='%s' for the simulation" % weight)
        print("Transactions simulated on original graph STARTED..")
        (
            shortest_paths,
            all_router_fees,
            attemp_times,
            success_rate,
        ) = get_shortest_paths(
            current_capacity_map,
            G,
            self.transactions,
            hash_transactions=with_node_removals,
            cost_prefix="original_",
            weight=weight,
            required_length=required_length,
            with_retry=with_retry,
        )
        print("Transactions simulated on original graph DONE")
        print("**************************************")
        print("Transaction succes rate:")
        print(success_rate)
        print("**************************************")
        return success_rate

    def all_path(
        self, src, trg, cutoff, threshold, G_for_path, num_paths=200, length=5
    ):
        # result = nx.flow.shortest_augmenting_path(G_for_path, src, trg)
        result = nx.maximum_flow(
            G_for_path,
            src,
            trg,
            capacity="capacity",
            flow_func=shortest_augmenting_path,
        )
        # result = nx.maximum_flow(G_for_path, src, trg, capacity="capacity")
        edge_lib = result[1]
        final = []
        open_path = []
        length = []

        for node, amount in edge_lib[src].items():
            if amount != 0:
                open_path.append([amount, src, node])
                edge_lib[src][node] = 0
        while len(open_path) != 0:
            # print(len(open_path))
            open_path_iter = open_path.pop(0)
            current = open_path_iter[-1]
            if current == trg:
                final.append(open_path_iter)
                continue
            for node, amount in edge_lib[current].items():
                if amount == 0:
                    continue
                elif amount >= open_path_iter[0]:
                    open_path_iter.append(node)
                    if node == trg:
                        final.append(open_path_iter)
                    else:
                        open_path.append(open_path_iter)
                    break
                elif amount < open_path_iter[0]:
                    residual = open_path_iter[0] - amount
                    open_path_iter[0] = residual
                    open_path.append(open_path_iter.copy())
                    open_path_iter[0] = amount
                    open_path_iter.append(node)
                    if node == trg:
                        final.append(open_path_iter)
                    else:
                        open_path.append(open_path_iter)
                    break
            edge_lib[current][open_path_iter[-1]] -= open_path_iter[0]
        for path in final:
            length.append(len(path) - 1)
        # print(np.mean(length))
        return final

    def get_hubs(self, G, percentage=0.01):
        hubs = dict()
        count = 0
        for node in list(G.nodes()):
            if "_trg" not in node:
                hubs[node] = G.degree(node, "capacity")
        hubs = sorted(hubs.items(), key=lambda item: item[1], reverse=True)
        hubs = list(hubs)
        end = int((len(hubs) * percentage))
        hubs = hubs[0:end]
        hubs_final = []
        for hub_record in hubs:
            hubs_final.append(hub_record[0])
        return hubs_final

    def analysis_path(self, hub_proportion):
        G = self.G_without_balance.copy()
        current_capacity_map = copy.deepcopy(self.current_capacity_map)
        lib = dict()
        lib_range = dict()
        path_percentage = dict()
        path_lib = dict()
        homedir = os.getcwd()
        valid_edge = dict()
        valid_edge_after_attack = dict()
        bound = dict()
        for key, value in current_capacity_map.items():
            lib[key] = value[3]
        L = sorted(lib.items(), key=lambda item: item[1])
        edge_sum = len(L)
        print_lib = dict()
        # for i in np.arange(100000, 2100000, 600000):
        #     valid_edge_after_attack[i] = [0, 0]
        #     channel_num = 0
        #     channel_capacity = 0
        #     for record in L:
        #         if record[1] > i:
        #             channel_num += 1
        #             channel_capacity += record[1]
        #     valid_edge[i] = [channel_num, channel_capacity]
        for i in range(0, 100, 10):
            print_lib[i] = set()
        # for record in L:
        classify_lib = dict()
        for i in range(0, 100, 10):
            start = int(i * edge_sum / 100)
            end = int((i + 10) * edge_sum / 100)
            for record in L[start:end]:
                classify_lib[record[0]] = i
        with open(
            homedir + "/ln_data/paths_" + str(hub_proportion) + ".csv",
            "r",
            encoding="utf-8",
        ) as f:
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

        for key in path_lib.keys():
            paths = path_lib[key]
            for path in paths:
                amount = float(path[0])
                path = path[1:]
                channels = [tuple(path[i : i + 2]) for i in range(len(path) - 1)]
                for channel in channels:
                    edge_class = classify_lib[channel]
                    print_lib[edge_class].add(channel)
        for i in range(0, 100, 10):
            print_lib[i] = len(print_lib[i]) / (0.1 * edge_sum)
        plt.figure(figsize=(10, 5), dpi=80)
        N = 10
        width = 0.35
        values = tuple(print_lib.values())
        index = np.arange(N)

    def analysis_path_fee(self, hub_proportion):
        G = self.G_without_balance.copy()
        current_capacity_map = copy.deepcopy(self.current_capacity_map)
        lib = dict()
        lib_range = dict()
        path_percentage = dict()
        path_lib = dict()
        homedir = os.getcwd()
        valid_edge = dict()
        valid_edge_after_attack = dict()
        bound = dict()
        for key, value in current_capacity_map.items():
            lib[key] = value[3]
        L = sorted(lib.items(), key=lambda item: item[1])
        edge_sum = len(L)
        print_lib = dict()
        channel_lib = set()
        with open(
            homedir + "/ln_data/paths_" + str(hub_proportion) + ".csv",
            "r",
            encoding="utf-8",
        ) as f:
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
        for key in path_lib.keys():
            paths = path_lib[key]
            for path in paths:
                amount = float(path[0])
                path = path[1:]
                for index in range(1, len(path) - 1):
                    channel_lib.add((path[index], path[index + 1]))
        channel_lib = list(channel_lib)
        return channel_lib

    def init_score_from_file(self, threshold, hub_proportion, count=-1):
        homedir = os.getcwd()
        path_lib = dict()
        score = dict()
        G = self.G_without_balance.copy()
        if hub_proportion == 0:
            self.score = []
            return
        with open(
            homedir + "/ln_data/paths_" + str(hub_proportion) + ".csv",
            "r",
            encoding="utf-8",
        ) as f:
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
        path_final = []

        for key, paths in path_lib.items():
            paths.sort(key=lambda x: len(x), reverse=True)
            amount_list = []
            for path in paths:
                amount = int(float(path[0]))
                path[0] = amount * (len(path) - 1)
                amount_list.append(amount)
                path_final.append(path)
        print("initialize score")
        self.score = sorted(path_final, key=lambda i: i[0], reverse=True)
        for pos in range(len(self.score)):
            self.score[pos] = self.score[pos][1:]
        return self.score

    def init_score_from_file_length(self, threshold, hub_proportion, count=-1):
        homedir = os.getcwd()
        path_lib = dict()
        score = dict()
        G = self.G_without_balance.copy()
        if hub_proportion == 0:
            self.score = []
            return
        with open(
            homedir + "/ln_data/paths_" + str(hub_proportion) + ".csv",
            "r",
            encoding="utf-8",
        ) as f:
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
        path_len = []
        path_final = []
        for key, paths in path_lib.items():
            score_iter = 0
            paths.sort(key=lambda x: len(x), reverse=True)
            amount_attack = 0
            amount_list = []
            for path in paths:
                amount = int(path[0])
                amount_list.append(amount)
                path = path[1:]
                path_len.append(len(path))
                path_final.append(path)
        print("initialize score")
        self.score = sorted(path_final, key=lambda i: len(i), reverse=True)
        return self.score

    def init_score_from_file_capacity(self, threshold, hub_proportion, count=-1):
        homedir = os.getcwd()
        path_lib = dict()
        score = dict()
        G = self.G_without_balance.copy()
        if hub_proportion == 0:
            self.score = []
            return
        with open(
            homedir + "/ln_data/paths_" + str(hub_proportion) + ".csv",
            "r",
            encoding="utf-8",
        ) as f:
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
        path_final = []
        for key, paths in path_lib.items():
            score_iter = 0
            paths.sort(key=lambda x: len(x), reverse=True)
            amount_attack = 0
            amount_list = []
            for path in paths:
                amount = int(path[0])
                path[0] = int(path[0])
                amount_list.append(amount)
                path_final.append(path)
        print("initialize score")
        self.score = sorted(path_final, key=lambda i: i[0], reverse=True)
        for pos in range(len(self.score)):
            self.score[pos] = self.score[pos][1:]
        return self.score

    def init_score_from_file_poor_node(self, threshold, hub_proportion, count=-1):
        homedir = os.getcwd()
        path_lib = dict()
        score = dict()
        G = self.G_without_balance.copy()
        res_sum = 0
        if hub_proportion == 0:
            self.score = []
            return
        with open(
            homedir + "/ln_data/paths_" + str(hub_proportion) + ".csv",
            "r",
            encoding="utf-8",
        ) as f:
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
        path_len = []
        path_final = []
        capacity = self.get_capacity(self.current_capacity_map)
        for key, paths in path_lib.items():
            paths.sort(key=lambda x: len(x), reverse=True)
            amount_attack = 0
            amount_list = []
            for path in paths:
                amount = int(path[0])
                path[0] = 0
                for pos in range(1, len(path) - 1):
                    # path[0] += (amount/balance[path[pos]] +
                    #             amount/balance[path[pos+1]])
                    capacity_A = capacity[path[pos]]
                    capacity_B = capacity[path[pos + 1]]
                    path[0] += amount / (
                        (capacity_A + 1) * (capacity_A + 1 - amount)
                    ) + amount / ((capacity_B + 1) * (capacity_B + 1 - amount))
                    # 1. amount 2. path_len 3. score 4. capacity
                path_final.append(path)
        print("initialize score")
        self.score = sorted(path_final, key=lambda i: i[0], reverse=True)
        for pos in range(len(self.score)):
            self.score[pos] = self.score[pos][1:]
        return self.score

    def init_score_from_file_fee(self, threshold, hub_proportion, count=-1):
        homedir = os.getcwd()
        path_lib = dict()
        score = dict()
        G = self.G_without_balance.copy()
        res_sum = 0
        if hub_proportion == 0:
            self.score = []
            return
        with open(
            homedir + "/ln_data/paths_" + str(hub_proportion) + ".csv",
            "r",
            encoding="utf-8",
        ) as f:
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
        path_len = []
        path_final = []
        capacity = self.get_capacity(self.current_capacity_map)
        for key, paths in path_lib.items():
            paths.sort(key=lambda x: len(x), reverse=True)
            amount_attack = 0
            amount_list = []
            for path in paths:
                amount = int(path[0])
                path[0] = 0
                for pos in range(1, len(path) - 1):
                    path[0] += G[path[pos]][path[pos + 1]]["total_fee"]
                path[0] = path[0] / (len(path) - 1)
                path_final.append(path)
        print("initialize score")
        self.score = sorted(path_final, key=lambda i: i[0], reverse=True)
        for pos in range(len(self.score)):
            self.score[pos] = self.score[pos][1:]
        return self.score

    def init_score(self, threshold, hub_proportion=0.01, count=-1):
        G = self.G_without_balance.copy()
        path_lib = dict()
        score = dict()
        delete_edge = []
        # get hubs, I select top 0.1% as hub. In this graph, there are 3 hubs
        hubs = self.get_hubs(G, hub_proportion)
        if hub_proportion == 0:
            self.score = []
            return
        total_path = 0
        attack_time = 0
        homedir = os.getcwd()
        print("Get %d hubs" % len(hubs))
        G_for_path = G.copy()
        for edge in G_for_path.edges():
            edge_begin, edge_end = edge
            if G_for_path[edge_begin][edge_end]["capacity"] < threshold:
                delete_edge.append((edge_begin, edge_end))
        G_for_path.remove_edges_from(delete_edge)
        pair_lib = dict()
        for (src, trg) in list(combinations(hubs, 2)):
            try:
                path = nx.shortest_path(G, source=src, target=trg)
                pair_lib[(src, trg)] = len(path)
            except nx.NetworkXNoPath:
                # if there is no path from src to trg
                pair_lib[(src, trg)] = 0
        pair_lib = sorted(pair_lib.items(), key=lambda item: item[1], reverse=True)
        node_set = set()
        for (src, trg), distance in pair_lib:
            path_lib[(src, trg)] = self.all_path(src, trg, 18, threshold, G_for_path)
            for path in path_lib[(src, trg)]:
                amount = path[0]
                path = path[1:]
                channels = [path[i : i + 2] for i in range(len(path) - 1)]
                for s, t in channels:
                    G_for_path[s][t]["capacity"] -= amount
                    if G_for_path[s][t]["capacity"] <= 0:
                        G_for_path.remove_edge(s, t)
            path_lib[(trg, src)] = self.all_path(trg, src, 18, threshold, G_for_path)
            for path in path_lib[(trg, src)]:
                amount = path[0]
                path = path[1:]
                channels = [path[i : i + 2] for i in range(len(path) - 1)]
                for s, t in channels:
                    G_for_path[s][t]["capacity"] -= amount
                    if G_for_path[s][t]["capacity"] <= 0:
                        G_for_path.remove_edge(s, t)
            print(
                "## There are %d paths between (%s - %s)."
                % (2 * len(path_lib[(src, trg)]), src, trg)
            )
            # accumulate total_path and attack_time
            attack_time += 2
            total_path += 2 * len(path_lib[(src, trg)])
        # store the paths
        with open(
            homedir + "/ln_data/paths_" + str(hub_proportion) + ".csv",
            "w",
            encoding="utf8",
        ) as f:
            w = csv.writer(f)
            for key in path_lib.keys():
                if len(path_lib[key]) == 0:
                    continue
                content = [key]
                # content.append(key)
                # for path in path_lib[key]:
                #     if path[0] == None:
                #         print("11")
                content.extend(path_lib[key])
                w.writerow(content)
            # w.writerows(path_lib.items())
        path_len = []
        for key, paths in path_lib.items():
            score_iter = 0
            paths.sort(key=lambda x: len(x), reverse=True)
            amount_attack = 0
            amount_list = []
            # edges_with capacity: 1. src 2. trg 3. capacity 4.fee
            # get edge with capacity
            # paths between two nodes
            for path in paths:
                amount = path[0]
                path = path[1:]
                score_iter += amount * (len(path) - 1)
                amount_attack += amount
                amount_list.append(amount)
            for path in paths:
                path.pop(0)
            score[key] = (score_iter, amount_attack, paths, amount_list)
        self.score = sorted(score.items(), key=lambda x: x[1][0], reverse=True)
        return self.score

    def amount_with_fee(self, path, amount_in_satoshi, fee_standard):
        sum_fee = 0
        fee_dict = dict()
        N = len(path)
        for i in range(N - 1):
            n1, n2 = path[i], path[i + 1]
            edge_record = fee_standard[(n1, n2)]
            fee_dict[(n1, n2)] = (
                edge_record["fee_base_msat"] / 1000.0
                + amount_in_satoshi * edge_record["fee_rate_milli_msat"] / 10.0 ** 6
            )
            fee_dict[(n1, n2)] = floor(fee_dict[(n1, n2)])
            sum_fee += fee_dict[(n1, n2)]
        return amount_in_satoshi + sum_fee

    def Dos_attack(self, edges, attack_cost, proportion=100):
        G = self.G_without_balance.copy()
        current_capacity_map = copy.deepcopy(self.current_capacity_map)
        with_depletion = True
        balance_total_now = 0
        balance_total = sum(i[0] for i in current_capacity_map.values())
        score_table = self.score
        total = sum(len(i[1][2]) for i in score_table)
        exhaust = False
        cost = 0
        count = 0
        fee_standard = dict()
        node_lib = set()
        attack_info = dict()
        count = 0
        # home_dir = homedir = "/Users/ZhiChunLu/research/Congestion/Congestion/Our/analysis/attack_info/"
        home_dir = "/Users/ZhiChunLu/research/Congestion/Congestion/Our/ln_data/"
        for index, row in edges.iterrows():
            key = (row["src"], row["trg"])
            fee_standard[key] = dict()
            fee_standard[key]["fee_base_msat"] = row["fee_base_msat"]
            fee_standard[key]["fee_rate_milli_msat"] = row["fee_rate_milli_msat"]
        for p in score_table:
            node_lib.add(p[0])
            node_lib.add(p[-1])
            if attack_cost == 0:
                break
                # check whether the edge has capacity larger than amount
            amount = current_capacity_map[(p[0], p[1])][0]
            for path_pos_iter in range(1, len(p) - 1):
                amount = min(
                    amount,
                    current_capacity_map[(p[path_pos_iter], p[path_pos_iter + 1])][0],
                )
            if attack_cost < 100:
                attack_cost = 0
            if attack_cost < amount:
                tx_amount = attack_cost
            else:
                tx_amount = amount
            if tx_amount == 0:
                continue
            # select the amount
            tx_amount_with_fee = self.amount_with_fee(p, tx_amount, fee_standard)
            tx_amount_init = tx_amount
            while True:
                if tx_amount_with_fee > attack_cost:
                    tx_amount -= 0.1 * tx_amount_init
                    tx_amount_with_fee = self.amount_with_fee(
                        p, tx_amount, fee_standard
                    )
                else:
                    break
            if tx_amount == 0:
                continue
            p[-1] = p[-1] + "_trg"
            while True:
                try:
                    process_path(
                        p,
                        tx_amount,
                        current_capacity_map,
                        self.G_without_balance,
                        "total_fee",
                        with_depletion,
                        Dos=True,
                        fee_standard=fee_standard,
                    )
                except RuntimeError as e:
                    tx_amount -= tx_amount_init * 0.1
                    if tx_amount <= 0:
                        tx_amount = 0
                    tx_amount_with_fee = self.amount_with_fee(
                        p, tx_amount, fee_standard
                    )
                    if tx_amount <= 0:
                        break
                    continue
                else:
                    channels = [p[i : i + 2] for i in range(len(p) - 1)]
                    for s, t in channels:
                        attack_info[count] = [
                            s,
                            t,
                            tx_amount_with_fee,
                            current_capacity_map[(s, t)][3],
                        ]
                        count += 1
                    attack_cost -= tx_amount_with_fee
                    break
            cost += tx_amount_with_fee
        record = [["Source", "Target", "attacked"]]
        record_set = set()
        for edge in G.edges():
            src, trg = edge
            if "_trg" in src:
                src = src.replace("_trg", "")
            if "_trg" in trg:
                trg = trg.replace("_trg", "")
            if (src, trg) in record_set or (trg, src) in record_set:
                continue
            else:
                record_set.add((src, trg))
            ratio = (
                1 - current_capacity_map[edge][0] / self.current_capacity_map[edge][0]
            )
            edge = (trg, src)
            if edge in G.edges():
                ratio2 = (
                    1
                    - current_capacity_map[edge][0] / self.current_capacity_map[edge][0]
                )
                # ratio2 = max(0.0001, ratio)
                ratio = np.mean([ratio, ratio2])
            if ratio == 0:
                ratio = "a"
            else:
                ratio = "b"
            # ratio = max(0.0001, ratio)
            G[src][trg]["ratio"] = ratio
            # replace _trg
            record.append([src, trg, ratio])
        with open(home_dir + "attack_overview.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(record)
        return cost

    def get_balance(self, current_capacity_map):
        account_balance = dict()
        for src, trg in current_capacity_map:
            value = current_capacity_map[(src, trg)][0]
            if src in account_balance.keys():
                account_balance[src] += value
            else:
                account_balance[src] = value
        return account_balance

    def get_capacity(self, current_capacity_map):
        account_balance = dict()
        for src, trg in current_capacity_map:
            value = current_capacity_map[(src, trg)][3]
            if src in account_balance.keys():
                account_balance[src] += value
            else:
                account_balance[src] = value
        return account_balance

    def simulate_Dos_attack(self, edges, attack_cost, bankrupt_amount):
        G = self.G.copy()
        bankrupt = 60000
        current_capacity_map = copy.deepcopy(self.current_capacity_map)
        account_balance_before = self.get_balance(current_capacity_map)
        with_depletion = True
        balance_total_now = 0
        balance_total = sum(i[0] for i in current_capacity_map.values())
        score_table = self.score
        total = sum(len(i[1][2]) for i in score_table)
        exhaust = False
        cost = 0
        count = 0
        fee_standard = dict()
        node_lib = set()
        for index, row in edges.iterrows():
            key = (row["src"], row["trg"])
            fee_standard[key] = dict()
            fee_standard[key]["fee_base_msat"] = row["fee_base_msat"]
            fee_standard[key]["fee_rate_milli_msat"] = row["fee_rate_milli_msat"]
        for p in score_table:
            node_lib.add(p[0])
            node_lib.add(p[-1])
            if attack_cost == 0:
                break
                # check whether the edge has capacity larger than amount
            amount = current_capacity_map[(p[0], p[1])][0]
            for path_pos_iter in range(1, len(p) - 1):
                amount = min(
                    amount,
                    current_capacity_map[(p[path_pos_iter], p[path_pos_iter + 1])][0],
                )
            if attack_cost < 100:
                attack_cost = 0
            if attack_cost < amount:
                tx_amount = attack_cost
            else:
                tx_amount = amount
            if tx_amount == 0:
                continue
            # select the amount
            tx_amount_with_fee = self.amount_with_fee(p, tx_amount, fee_standard)
            tx_amount_init = tx_amount
            while True:
                if tx_amount_with_fee > attack_cost:
                    tx_amount -= 0.1 * tx_amount_init
                    tx_amount_with_fee = self.amount_with_fee(
                        p, tx_amount, fee_standard
                    )
                else:
                    break
            if tx_amount == 0:
                continue
            p[-1] = p[-1] + "_trg"
            while True:
                try:
                    process_path(
                        p,
                        tx_amount,
                        current_capacity_map,
                        self.G_without_balance,
                        "total_fee",
                        with_depletion,
                        Dos=True,
                        fee_standard=fee_standard,
                    )
                except RuntimeError as e:
                    tx_amount -= tx_amount_init * 0.1
                    if tx_amount <= 0:
                        tx_amount = 0
                    tx_amount_with_fee = self.amount_with_fee(
                        p, tx_amount, fee_standard
                    )
                    if tx_amount <= 0:
                        break
                    continue
                else:
                    attack_cost -= tx_amount_with_fee
                    break
            cost += tx_amount_with_fee
        balance_total_now = sum(i[0] for i in current_capacity_map.values())
        locked = balance_total - balance_total_now
        cost_ratio = cost / balance_total
        for key in current_capacity_map.keys():
            if current_capacity_map[key][0] > self.current_capacity_map[key][0]:
                print("wrong")
        lock_ratio = locked / balance_total

        print(lock_ratio)

    def simulate_with_Dos_attack(self, edges, attack_cost, bankrupt_amount):
        G = self.G.copy()
        current_capacity_map = copy.deepcopy(self.current_capacity_map)
        account_balance_before = self.get_balance(current_capacity_map)
        with_depletion = True
        balance_total_now = 0
        balance_total = sum(i[0] for i in current_capacity_map.values())
        score_table = self.score
        total = sum(len(i[1][2]) for i in score_table)
        exhaust = False
        cost = 0
        count = 0
        fee_standard = dict()
        node_lib = set()
        for index, row in edges.iterrows():
            key = (row["src"], row["trg"])
            fee_standard[key] = dict()
            fee_standard[key]["fee_base_msat"] = row["fee_base_msat"]
            fee_standard[key]["fee_rate_milli_msat"] = row["fee_rate_milli_msat"]
        for p in score_table:
            node_lib.add(p[0])
            node_lib.add(p[-1])
            if attack_cost == 0:
                break
                # check whether the edge has capacity larger than amount
            amount = current_capacity_map[(p[0], p[1])][0]
            for path_pos_iter in range(1, len(p) - 1):
                amount = min(
                    amount,
                    current_capacity_map[(p[path_pos_iter], p[path_pos_iter + 1])][0],
                )
            if attack_cost < 100:
                attack_cost = 0
            if attack_cost < amount:
                tx_amount = attack_cost
            else:
                tx_amount = amount
            if tx_amount == 0:
                continue
            # select the amount
            tx_amount_with_fee = self.amount_with_fee(p, tx_amount, fee_standard)
            tx_amount_init = tx_amount
            while True:
                if tx_amount_with_fee > attack_cost:
                    tx_amount -= 0.1 * tx_amount_init
                    tx_amount_with_fee = self.amount_with_fee(
                        p, tx_amount, fee_standard
                    )
                else:
                    break
            if tx_amount == 0:
                continue
            p[-1] = p[-1] + "_trg"
            while True:
                try:
                    process_path(
                        p,
                        tx_amount,
                        current_capacity_map,
                        self.G_without_balance,
                        "total_fee",
                        with_depletion,
                        Dos=True,
                        fee_standard=fee_standard,
                    )
                except RuntimeError as e:
                    tx_amount -= tx_amount_init * 0.1
                    if tx_amount <= 0:
                        tx_amount = 0
                    tx_amount_with_fee = self.amount_with_fee(
                        p, tx_amount, fee_standard
                    )
                    if tx_amount <= 0:
                        break
                    continue
                else:
                    attack_cost -= tx_amount_with_fee
                    break
            cost += tx_amount_with_fee
        balance_total_now = sum(i[0] for i in current_capacity_map.values())
        locked = balance_total - balance_total_now
        cost_ratio = cost / balance_total
        # if spent money reaches the threshold
        # stop attacking and check liquidity
        for key in current_capacity_map.keys():
            if current_capacity_map[key][0] > self.current_capacity_map[key][0]:
                print("wrong")
        lock_ratio = locked / balance_total

        print(lock_ratio)

        account_balance_after = self.get_balance(current_capacity_map)
        bankrupt_num = sum(
            balance < bankrupt_amount for balance in account_balance_after.values()
        )
        print("after attack, # bankrupt nodes.", bankrupt_num)
        print("Transactions simulated on original graph STARTED..")
        (
            avg_fee,
            avg_attempt,
            length_lib,
            success_rate,
            path_info,
        ) = get_shortest_paths(
            current_capacity_map,
            G,
            self.transactions,
            hash_transactions=True,
            cost_prefix="original_",
            weight="total_fee",
            required_length=None,
            org_capacity=self.current_capacity_map,
            with_retry=True,
        )
        print("Transactions simulated on original graph DONE")
        print("**************************************")
        print("Transaction succes rate:")
        print(success_rate)
        print("**************************************")

        account_balance_after = self.get_balance(current_capacity_map)
        bankrupt_num = sum(
            balance < bankrupt_amount for balance in account_balance_after.values()
        )
        print("after payments, # bankrupt nodes.", bankrupt_num)

        return (
            cost,
            cost_ratio,
            locked,
            lock_ratio,
            success_rate,
            avg_fee,
            avg_attempt,
            length_lib,
            len(node_lib),
            path_info,
            bankrupt_num,
        )

    def export(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open("%s/params.json" % output_dir, "w") as fp:
            json.dump(self.params, fp)
        length_distrib = self.shortest_paths["length"].value_counts()
        length_distrib.to_csv("%s/lengths_distrib.csv" % output_dir)
        total_income = get_total_income_for_routers(self.all_router_fees)
        total_income.to_csv("%s/router_incomes.csv" % output_dir, index=False)
        total_fee = get_total_fee_for_sources(self.transactions, self.shortest_paths)
        total_fee.to_csv("%s/source_fees.csv" % output_dir, index=True)
        print("Export DONE")
        return total_income, total_fee

    def remove_edge(self, src, trg):
        index1 = self.edges[
            (self.edges["src"] == src) & (self.edges["trg"] == trg)
        ].index
        index2 = self.edges[
            (self.edges["src"] == trg) & (self.edges["trg"] == src)
        ].index
        self.edges = self.edges.drop(index1)
        self.edges = self.edges.drop(index2)


### process results ###


def get_total_income_for_routers(all_router_fees):
    grouped = all_router_fees.groupby("node")
    aggr_router_income = (
        grouped.agg({"fee": "sum", "transaction_id": "count"})
        .reset_index()
        .sort_values("fee", ascending=False)
    )
    return aggr_router_income.rename({"transaction_id": "num_trans"}, axis=1)


def get_total_fee_for_sources(transactions, shortest_paths):
    tmp_sp = shortest_paths[shortest_paths["length"] > 0]
    trans_with_costs = transactions[["transaction_id", "source"]].merge(
        tmp_sp[["transaction_id", "original_cost"]], on="transaction_id", how="right"
    )
    agg_funcs = dict(original_cost="mean", transaction_id="count")
    aggs = (
        trans_with_costs.groupby(by="source")["original_cost"]
        .agg(agg_funcs)
        .rename({"original_cost": "mean_fee", "transaction_id": "num_trans"}, axis=1)
    )
    return aggs


### optimal fee pricing ###


def inspect_base_fee_thresholds(ordered_deltas, pos_thresholds, min_ratio):
    thresholds = [0.0] + pos_thresholds
    original_income = ordered_deltas["fee"].sum()
    original_num_transactions = len(ordered_deltas)
    incomes, probas = [original_income], [1.0]
    # inspect only positive deltas
    for th in thresholds[1:]:
        # transactions that will still pay the increased base_fee
        df = ordered_deltas[ordered_deltas["delta_cost"] >= th]
        prob = len(df) / original_num_transactions
        probas.append(prob)
        # adjusted router income at the new threshold
        adj_income = df["fee"].sum() + len(df) * th
        incomes.append(adj_income)
        if prob < min_ratio:
            break
    return incomes, probas, thresholds, original_income, original_num_transactions


def visualize_thresholds(incomes, probas, thresholds, original_num_transactions):
    fig, ax1 = plt.subplots()
    x = thresholds[: len(incomes)]
    ax1.set_title(original_num_transactions)
    ax1.plot(x, incomes, "bx-")
    ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.plot(x, probas, "gx-")
    ax2.set_xscale("log")


def calculate_max_income(
    n, p_altered, shortest_paths, all_router_fees, visualize=False, min_ratio=0.0
):
    trans = p_altered[p_altered["node"] == n]
    trans = trans.merge(
        shortest_paths[["transaction_id", "original_cost"]],
        on="transaction_id",
        how="inner",
    )
    # 'fee' column is merged
    trans = trans.merge(all_router_fees, on=["transaction_id", "node"], how="inner")
    # router could ask for this cost difference
    trans["delta_cost"] = trans["cost"] - trans["original_cost"]
    ordered_deltas = trans[["transaction_id", "fee", "delta_cost"]].sort_values(
        "delta_cost"
    )
    ordered_deltas["delta_cost"] = ordered_deltas["delta_cost"].apply(
        lambda x: round(x, 2)
    )
    pos_thresholds = sorted(
        list(ordered_deltas[ordered_deltas["delta_cost"] > 0.0]["delta_cost"].unique())
    )
    (
        incomes,
        probas,
        thresholds,
        alt_income,
        alt_num_trans,
    ) = inspect_base_fee_thresholds(ordered_deltas, pos_thresholds, min_ratio)
    if visualize:
        visualize_thresholds(incomes, probas, thresholds, alt_num_trans)
    max_idx = np.argmax(incomes)
    return (
        thresholds[max_idx],
        incomes[max_idx],
        probas[max_idx],
        alt_income,
        alt_num_trans,
    )


def calc_optimal_base_fee(shortest_paths, alternative_paths, all_router_fees):
    # paths with length at least 2
    valid_sp = shortest_paths[shortest_paths["length"] > 1]
    # drop failed alternative paths
    p_altered = alternative_paths[~alternative_paths["cost"].isnull()]
    num_routers = len(alternative_paths["node"].unique())
    num_routers_with_alternative_paths = len(p_altered["node"].unique())
    routers = list(p_altered["node"].unique())
    opt_strategy = []
    for n in tqdm(routers, mininterval=5):
        (
            opt_delta,
            opt_income,
            opt_ratio,
            origi_income,
            origi_num_trans,
        ) = calculate_max_income(
            n, p_altered, valid_sp, all_router_fees, visualize=False
        )
        opt_strategy.append(
            (n, opt_delta, opt_income, opt_ratio, origi_income, origi_num_trans)
        )
    opt_fees_df = pd.DataFrame(
        opt_strategy,
        columns=[
            "node",
            "opt_delta",
            "opt_alt_income",
            "opt_alt_traffic",
            "alt_income",
            "alt_traffic",
        ],
    )
    total_income = get_total_income_for_routers(all_router_fees).rename(
        {"fee": "total_income", "num_trans": "total_traffic"}, axis=1
    )
    merged_infos = total_income.merge(opt_fees_df, on="node", how="outer")
    merged_infos = merged_infos.sort_values("total_income", ascending=False)
    merged_infos = merged_infos.fillna(0.0)
    merged_infos["failed_traffic"] = (
        merged_infos["total_traffic"] - merged_infos["alt_traffic"]
    )
    merged_infos["failed_traffic_ratio"] = (
        merged_infos["failed_traffic"] / merged_infos["total_traffic"]
    )
    merged_infos["income_diff"] = merged_infos.apply(
        lambda x: x["opt_alt_income"]
        - x["alt_income"]
        + x["failed_traffic"] * x["opt_delta"],
        axis=1,
    )
    merged_infos
    return (
        merged_infos[
            [
                "node",
                "total_income",
                "total_traffic",
                "failed_traffic_ratio",
                "opt_delta",
                "income_diff",
            ]
        ],
        p_altered,
    )
