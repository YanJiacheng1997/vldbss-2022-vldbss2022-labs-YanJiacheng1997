import evaluation_utils as eval_utils
import matplotlib.pyplot as plt
import numpy as np
import range_query as rq
import json
import torch
import torch.nn as nn
import statistics as stats
import xgboost as xgb


def min_max_normalize(v, min_v, max_v):
    assert max_v > min_v
    return (v-min_v)/(max_v-min_v)


def extract_features_from_query(range_query, table_stats, considered_cols):
    # feat:     [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
    #           <-                   range features                    ->, <-     est features     ->
    feature = []
    for col in considered_cols:
        min_val = table_stats.columns[col].min_val()
        max_val = table_stats.columns[col].max_val()
        (left, right) = range_query.column_range(col, min_val, max_val)
        left = min_max_normalize(left, min_val, max_val)
        right = min_max_normalize(right, min_val, max_val)
        feature.append(left)
        feature.append(right)
    feature.append(stats.AVIEstimator.estimate(range_query, table_stats))
    feature.append(stats.ExpBackoffEstimator.estimate(range_query, table_stats))
    feature.append(stats.MinSelEstimator.estimate(range_query, table_stats))
    return feature


def preprocess_queries(queris, table_stats, columns):
    inputs, labels = [], []
    for item in queris:
        query = rq.ParsedRangeQuery.parse_range_query(item['query'])
        feats = extract_features_from_query(query, table_stats, columns)
        inputs.append(feats)
        labels.append(np.log2(item['act_rows']))
    return inputs, labels


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, query_data, stats, columns):
        super().__init__()
        self.train_data = []
        for item in query_data:
            query = rq.ParsedRangeQuery.parse_range_query(item['query'])
            # print(f"query={query}")
            feats = extract_features_from_query(query, stats, columns)
            # feature = torch.Tensor(feats[:-3])
            feature = torch.Tensor(feats)
            # print(f"feature={feature}")
            # # We want the model to predict log2(est_rows) rather than est_rows since est_rows must be large than or
            # # equal to 0 while log2(est_rows) don't have any boundary. Hence we use log2(act_rows) as label rather
            # # than act_rows.
            # label = torch.log2(torch.Tensor([item['act_rows']]))
            label = torch.Tensor([item['act_rows']])
            self.train_data.append((feature, label))

    def __getitem__(self, index):
        return self.train_data[index]

    def __len__(self):
        return len(self.train_data)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(15, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                m.weight.data.uniform_(-0.0001, 0.0001)
                m.bias.data.fill_(0)
        self.apply(init_fn)


def est_mlp(train_data, test_data, table_stats, columns):
    train_dataset = QueryDataset(train_data, table_stats, columns)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)

    mlp = MLP()
    mlp.init_weights()

    def loss_fn(predict_rows, actual_rows):
        # return torch.mean(torch.square(torch.log2(torch.abs(predict_rows)) - torch.log2(actual_rows)))
        est_rows = torch.abs(predict_rows)
        return torch.mean(torch.square((est_rows - actual_rows) / (est_rows + actual_rows)))
        # return torch.mean(torch.square(est_rows / actual_rows) + torch.square(actual_rows / est_rows))
        # x = torch.clamp(torch.abs(predict_rows), min=1)
        # y = actual_rows
        # qerror = torch.max(x / y, y / x)
        # return torch.mean(torch.square(qerror))

    # loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    num_epoch = 10
    for epoch in range(num_epoch):
        print(f"epoch {epoch} start")
        total_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = mlp(inputs)
            # print(f"inputs={inputs}")
            # print(f"outputs={outputs}")
            # print(f"labels={labels}")
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # print(f"i={i}, loss={loss.item()}")
            # if i % 100 == 0:
            #     print(f"i={i}, loss={loss.item()}")
            total_loss += loss.item()
        print(f"epoch {epoch} finish, total_loss={format(total_loss, '.10E')}")

    train_est_rows, train_act_rows = [], []
    for i, data in enumerate(train_loader):
        inputs, labels = data
        outputs = mlp(inputs)
        est = torch.abs(outputs).tolist()
        # est = outputs.tolist()
        act = labels.tolist()
        # est = torch.exp2(outputs).tolist()
        # act = torch.exp2(labels).tolist()
        # est = torch.clamp(torch.abs(outputs), min=1).tolist()
        # act = labels.tolist()
        print(f"i={i}, est={est}, act={act}")
        train_est_rows.extend(est)
        train_act_rows.extend(act)

    test_dataset = QueryDataset(test_data, table_stats, columns)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)

    test_est_rows, test_act_rows = [], []
    for i, data in enumerate(test_loader):
        inputs, labels = data
        outputs = mlp(inputs)
        est = torch.abs(outputs).tolist()
        # est = outputs.tolist()
        act = labels.tolist()
        # est = torch.exp2(outputs).tolist()
        # act = torch.exp2(labels).tolist()
        # est = torch.clamp(torch.abs(outputs), min=1).tolist()
        # act = labels.tolist()
        print(f"i={i}, est={est}, act={act}")
        test_est_rows.extend(est)
        test_act_rows.extend(act)

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def est_xgb(train_data, test_data, table_stats, columns):
    print("estimate row counts by xgboost")
    train_x, train_y = preprocess_queries(train_data, table_stats, columns)
    regressor = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4)
    print('start fitting')
    regressor.fit(train_x, train_y)
    print('finish fitting')
    train_pred = regressor.predict(train_x)
    train_est_rows = np.exp2(train_pred).tolist()
    train_act_rows = np.exp2(train_y).tolist()

    test_x, test_y = preprocess_queries(test_data, table_stats, columns)
    test_pred = regressor.predict(test_x)
    test_est_rows = np.exp2(test_pred).tolist()
    test_act_rows = np.exp2(test_y).tolist()

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def eval_model(model, train_data, test_data, table_stats, columns):
    if model == 'mlp':
        est_fn = est_mlp
    else:
        est_fn = est_xgb

    train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_fn(train_data, test_data, table_stats, columns)

    name = f'{model}_train_{len(train_data)}'
    eval_utils.draw_act_est_figure(name, train_act_rows, train_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(train_act_rows, train_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')

    name = f'{model}_test_{len(test_data)}'
    eval_utils.draw_act_est_figure(name, test_act_rows, test_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(test_act_rows, test_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')


if __name__ == '__main__':
    stats_json_file = './data/title_stats.json'
    train_json_file = './data/query_train_20000.json'
    test_json_file = './data/query_test_5000.json'
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)

    eval_model('xgb', train_data, test_data, table_stats, columns)

