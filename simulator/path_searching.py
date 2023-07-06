import copy
import datetime
import re
from collections import Counter
from math import *

import networkx as nx
import numpy as np
import pandas as pd

from .genetic_routing import GeneticPaymentRouter


def get_balance(current_capacity_map):
    account_balance = dict()
    for src, trg in current_capacity_map:
        value = current_capacity_map[(src, trg)][0]
        if src in account_balance.keys():
            account_balance[src] += value
        else:
            account_balance[src] = value
    return account_balance


def get_shortest_paths(
    init_capacities,
    G_origi,
    transactions,
    hash_transactions=True,
    cost_prefix="",
    weight="total_fee",
    required_length=None,
    org_capacity=None,
    with_retry=False,
):
    G = G_origi.copy()  # copy due to forthcoming graph capacity changes!!!
    # capacity_map = copy.deepcopy(init_capacities)
    capacity_map = init_capacities
    with_depletion = capacity_map != None
    shortest_paths = []
    total_depletions = dict()
    router_fee_tuples = []
    attemp_times = []
    success_id = set()
    genetic_rounds = []
    fee_lib = dict()
    attemp_times = dict()
    length_lib = dict()
    path_info = dict()
    for idx, row in transactions.iterrows():
        p, cost = [], None
        tmp_remove = []
        attemp_time = 1
        fee_value = None
        path_info[idx] = list()

        try:
            S, T = row["source"], row["target"] + "_trg"
            # if src and trg in the nodes
            if (not S in G.nodes()) or (not T in G.nodes()):
                shortest_paths.append((row["transaction_id"], cost, len(p) - 1, p))
                continue
            while True:
                try:
                    p = nx.shortest_path(G, source=S, target=T, weight=weight)
                    p_append = []
                    for node in p:
                        if "_trg" in node:
                            node = node.replace("_trg", "")
                        p_append.append(node)
                    path_info[idx].append(p_append)
                    fee_value, router_fees, depletions, len_p = process_path(
                        p,
                        row["amount_SAT"],
                        capacity_map,
                        G,
                        "total_fee",
                        with_depletion,
                    )
                except nx.NetworkXNoPath:
                    raise
                except RuntimeError as e:
                    attemp_time += 1
                    if attemp_time > 10:
                        raise
                    # remove path without capacity temporarily
                    if with_retry:
                        src_trg = str(e).split(":")[1]
                        src_remove = src_trg.split("-")[0]
                        src_remove = src_remove[1:]
                        trg_remove = src_trg.split("-")[1]
                        capacity_remove = G[src_remove][trg_remove]["capacity"]
                        total_fee_remove = G[src_remove][trg_remove]["total_fee"]
                        G.remove_edge(src_remove, trg_remove)
                        tmp_remove.append(
                            (src_remove, trg_remove, capacity_remove, total_fee_remove)
                        )
                        (cap, fee, is_trg, total_cap) = capacity_map[
                            (src_remove, trg_remove)
                        ]
                        if is_trg:
                            G.remove_edge(src_remove, trg_remove + "_trg")
                            tmp_remove.append(
                                (src_remove, trg_remove + "_trg", 0, total_fee_remove)
                            )
                        continue
                    raise
                else:
                    length_lib[idx] = len_p
                    fee_lib[idx] = fee_value
                    attemp_times[idx] = attemp_time
                    # attemp_times.append(attemp_time)
                    # fee_lib.append(fee_value)
                    break
            # recover edges
            # for iter_record in tmp_remove:
            #     G.add_weighted_edges_from([iter_record], weight="total_fee")
            if row["target"] in p:
                raise RuntimeError("Loop detected: %s" % row["target"])
            routers = list(router_fees.keys())

            # id; node; fee;
            router_fee_tuples += list(
                zip(
                    [row["transaction_id"]] * len(router_fees),
                    router_fees.keys(),
                    router_fees.values(),
                )
            )
        except nx.NetworkXNoPath:
            continue
        except RuntimeError:
            continue
        finally:
            if with_retry:
                for record in tmp_remove:
                    src_recover, trg_recover, capa_recover, fee_recover = record
                    G.add_edge(
                        src_recover,
                        trg_recover,
                        capacity=capa_recover,
                        total_fee=fee_recover,
                    )
            shortest_paths.append((row["transaction_id"], fee_value, len(p) - 1, p))
    all_router_fees = pd.DataFrame(
        router_fee_tuples, columns=["transaction_id", "node", "fee"]
    )
    for path in shortest_paths:
        if path[1] != None and path[2] < 18:
            success_id.add(path[0])
    success_rate = len(success_id) / len(transactions)
    return fee_lib, attemp_times, length_lib, success_rate, path_info


def process_path(
    path,
    amount_in_satoshi,
    capacity_map,
    G,
    weight,
    with_depletion,
    Dos=False,
    fee_standard=None,
):
    routers = {}
    depletions = []
    sum_fee = 0
    N = len(path)
    path[-1] = path[N - 1].replace("_trg", "")
    fee_dict = dict()
    if Dos:
        # get sum_fee
        for i in range(N - 1):
            n1, n2 = path[i], path[i + 1]
            edge_record = fee_standard[(n1, n2)]
            fee_dict[(n1, n2)] = (
                edge_record["fee_base_msat"] / 1000.0
                + amount_in_satoshi * edge_record["fee_rate_milli_msat"] / 10.0 ** 6
            )
            fee_dict[(n1, n2)] = floor(fee_dict[(n1, n2)])
            sum_fee += fee_dict[(n1, n2)]
        for i in range(N - 1):
            n1, n2 = path[i], path[i + 1]
            sum_fee -= fee_dict[(n1, n2)]
            judge_forward_edge(capacity_map, G, amount_in_satoshi, sum_fee, n1, n2)
    else:
        for i in range(1, N - 1):
            n1, n2 = path[i], path[i + 1]
            fee_dict[(n1, n2)] = G[n1][n2][weight]
            fee_dict[(n1, n2)] = floor(fee_dict[(n1, n2)])
            sum_fee += fee_dict[(n1), (n2)]
        n1, n2 = path[0], path[1]
        judge_forward_edge(capacity_map, G, amount_in_satoshi, sum_fee, n1, n2)
        for i in range(1, N - 1):
            n1, n2 = path[i], path[i + 1]
            sum_fee -= fee_dict[(n1, n2)]
            judge_forward_edge(capacity_map, G, amount_in_satoshi, sum_fee, n1, n2)

    # if the path is viable, update the balance
    sum_fee = sum(fee_dict.values())
    if Dos:
        for i in range(N - 1):
            n1, n2 = path[i], path[i + 1]
            sum_fee -= fee_dict[(n1, n2)]
            routers[n1] = fee_dict[(n1, n2)]
            if with_depletion:
                # whether n1-n2 is depleted
                n1_removed = process_forward_edge(
                    capacity_map, G, amount_in_satoshi, sum_fee, n1, n2, Dos
                )
                if n1_removed:
                    depletions.append(n1)
    else:
        n1, n2 = path[0], path[1]
        if with_depletion:
            n1_removed = process_forward_edge(
                capacity_map, G, amount_in_satoshi, sum_fee, n1, n2, Dos
            )
            if n1_removed:
                depletions.append(n1)
            process_backward_edge(
                capacity_map, G, amount_in_satoshi, sum_fee, n2, n1, Dos
            )
        for i in range(1, N - 1):
            n1, n2 = path[i], path[i + 1]
            sum_fee -= fee_dict[(n1, n2)]
            routers[n1] = fee_dict[(n1, n2)]
            if with_depletion:
                # whether n1-n2 is depleted
                n1_removed = process_forward_edge(
                    capacity_map, G, amount_in_satoshi, sum_fee, n1, n2, Dos
                )
                if n1_removed:
                    depletions.append(n1)
                process_backward_edge(
                    capacity_map, G, amount_in_satoshi, sum_fee, n2, n1, Dos
                )
    if not Dos:
        path[N - 1] = path[N - 1] + "_trg"
    return np.sum(list(routers.values())), routers, depletions, len(path)


def judge_forward_edge(capacity_map, G, amount_in_satoshi, sum_fee, src, trg):
    cap, fee, is_trg, total_cap = capacity_map[(src, trg)]
    if cap < amount_in_satoshi + sum_fee:
        raise RuntimeError("forward %i: %s-%s" % (cap, src, trg))


def process_forward_edge(capacity_map, G, amount_in_satoshi, sum_fee, src, trg, Dos):
    # if no enough capacity, throw runtime error, else if can not route anymore
    # just delete the edge
    removed = False
    cap, fee, is_trg, total_cap = capacity_map[(src, trg)]
    if cap < 2 * amount_in_satoshi + sum_fee:  # cannot route more transactions
        if not Dos:
            G.remove_edge(src, trg)
            if is_trg:
                G.remove_edge(src, trg + "_trg")
    capacity_map[(src, trg)] = [
        cap - amount_in_satoshi - sum_fee,
        fee,
        is_trg,
        total_cap,
    ]
    return removed


def process_backward_edge(capacity_map, G, amount_in_satoshi, sum_fee, src, trg, Dos):
    if (src, trg) in capacity_map:
        cap, fee, is_trg, total_cap = capacity_map[(src, trg)]
        if not Dos:
            capacity_map[(src, trg)] = [
                cap + amount_in_satoshi + sum_fee,
                fee,
                is_trg,
                total_cap,
            ]
            if cap < amount_in_satoshi:  # it can route transactions again
                G.add_edge(
                    src, trg, capacity=cap + amount_in_satoshi + sum_fee, total_fee=fee
                )
                if is_trg:
                    G.add_edge(src, trg + "_trg", capacity=0, total_fee=fee)
