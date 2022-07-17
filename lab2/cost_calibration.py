import math

import numpy as np
from sklearn.linear_model import LinearRegression


def estimate_plan(operator, factors, weights):
    cost = 0.0
    for child in operator.children:
        cost += estimate_plan(child, factors, weights)

    if operator.is_hash_agg():
        # hash_agg_cost = input_row_cnt * cpu_fac
        input_row_cnt = operator.children[0].est_row_counts()
        cost += input_row_cnt * factors['cpu']
        weights['cpu'] += input_row_cnt

    elif operator.is_hash_join():
        # hash_join_cost = (build_hashmap_cost + probe_and_pair_cost)
        #   = (build_row_cnt * cpu_fac) + (output_row_cnt * cpu_fac)
        output_row_cnt = operator.est_row_counts()
        build_side = int(operator.children[1].is_build_side())
        build_row_cnt = operator.children[build_side].est_row_counts()

        cost += (build_row_cnt + output_row_cnt) * factors['cpu']
        weights['cpu'] += (build_row_cnt + output_row_cnt)

    elif operator.is_sort():
        # sort_cost = input_row_cnt * log(input_row_cnt) * cpu_fac
        input_row_cnt = operator.children[0].est_row_counts()
        cost += input_row_cnt * math.log2(max(input_row_cnt, 1)) * factors['cpu']
        weights['cpu'] += input_row_cnt * math.log2(max(input_row_cnt, 1))

    elif operator.is_selection():
        # selection_cost = input_row_cnt * cpu_fac
        input_row_cnt = operator.children[0].est_row_counts()
        cost += input_row_cnt * factors['cpu']
        weights['cpu'] += input_row_cnt

    elif operator.is_projection():
        # projection_cost = input_row_cnt * cpu_fac
        input_row_cnt = operator.children[0].est_row_counts()
        cost += input_row_cnt * factors['cpu']
        weights['cpu'] += input_row_cnt

    elif operator.is_table_reader():
        # table_reader_cost = input_row_cnt * input_row_size * net_fac
        input_row_cnt = operator.children[0].est_row_counts()
        input_row_size = operator.children[0].row_size()
        cost += input_row_cnt * input_row_size * factors['net']
        weights['net'] += input_row_cnt * input_row_size

    elif operator.is_table_scan():
        # table_scan_cost = row_cnt * row_size * scan_fac
        row_cnt = operator.est_row_counts()
        row_size = operator.row_size()
        cost += row_cnt * row_size * factors['scan']
        weights['scan'] += row_cnt * row_size

    elif operator.is_index_reader():
        # index_reader_cost = input_row_cnt * input_row_size * net_fac
        input_row_cnt = operator.children[0].est_row_counts()
        input_row_size = operator.children[0].row_size()
        cost += input_row_cnt * input_row_size * factors['net']
        weights['net'] += input_row_cnt * input_row_size

    elif operator.is_index_scan():
        # index_scan_cost = row_cnt * row_size * scan_fac
        row_cnt = operator.est_row_counts()
        row_size = operator.row_size()
        cost += row_cnt * row_size * factors['scan']
        weights['scan'] += row_cnt * row_size

    elif operator.is_index_lookup():
        # index_lookup_cost = net_cost + seek_cost
        #   = (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size) * net_fac +
        #     (build_row_cnt / batch_size) * seek_fac
        build_side = int(operator.children[1].is_build_side())
        build_row_cnt = operator.children[build_side].est_row_counts()
        build_row_size = operator.children[build_side].row_size()
        probe_row_cnt = operator.children[1 - build_side].est_row_counts()
        probe_row_size = operator.children[1 - build_side].row_size()
        batch_size = operator.batch_size()

        cost += (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size) * factors['net']
        weights['net'] += (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size)

        cost += (build_row_cnt / batch_size) * factors['seek']
        weights['seek'] += (build_row_cnt / batch_size)

    else:
        print(operator.id)
        assert (1 == 2)  # unknown operator
    return cost


def estimate_calibration(train_plans, test_plans):
    # init factors
    factors = {
        "cpu": 1,
        "scan": 1,
        "net": 1,
        "seek": 1,
    }

    # get training data: factor weights and act_time
    est_costs_before = []
    act_times = []
    weights = []
    for p in train_plans:
        w = {"cpu": 0, "scan": 0, "net": 0, "seek": 0}
        cost = estimate_plan(p.root, factors, w)
        weights.append(w)
        act_times.append(p.exec_time_in_ms())
        est_costs_before.append(cost)

    # training: factors * weights = act_time
    x = []
    y = []
    for i in range(0, len(act_times)):
        x.append([weights[i]["cpu"], weights[i]["scan"],
                  weights[i]["net"], weights[i]["seek"]])
        y.append(act_times[i] * 1000000)
    nx = np.array(x)
    ny = np.array(y)
    # TODO: normalization
    lr = LinearRegression().fit(nx, ny)
    factors = {
        "cpu": lr.coef_[0],
        "scan": lr.coef_[1],
        "net": lr.coef_[2],
        "seek": lr.coef_[3],
    }
    print("--->>> regression cost factors: ", factors)

    # evaluation
    est_costs = []
    for p in test_plans:
        w = {"cpu": 0, "scan": 0, "net": 0, "seek": 0}
        cost = estimate_plan(p.root, factors, w)
        est_costs.append(cost)

    return est_costs
