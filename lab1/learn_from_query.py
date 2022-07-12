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
    # The function may be useful when dealing with lower/upper bounds of columns.
    assert max_v > min_v
    return (v-min_v)/(max_v-min_v)


def extract_features_from_query(range_query, table_stats, considered_cols):
    # feat:     [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
    #           <-                   range features                    ->, <-     est features     ->
    feature = []
    # YOUR CODE HERE: extract features from query
    for col in considered_cols:
        min_val = table_stats.columns[col].min_val()
        max_val = table_stats.columns[col].max_val()
        left,right = range_query.column_range(col,min_val,max_val)
        feature.append(min_max_normalize(left,min_val,max_val))
        feature.append(min_max_normalize(right,min_val,max_val))
        feature.append(stats.MinSelEstimator.estimate((range_query,table_stats)))
    return feature



def preprocess_queries(queris, table_stats, columns):
    """
    preprocess_queries turn queries into features and labels, which are used for regression model.
    """
    features, labels = [], []
    for item in queris:
        query, act_rows = item['query'], item['act_rows']
        feature, label = None, None
        # YOUR CODE HERE: transform (query, act_rows) to (feature, label)
        # Some functions like rq.ParsedRangeQuery.parse_range_query and extract_features_from_query may be helpful.
        parsed = rq.ParsedRangeQuery.parse_range_query((query))
        feature = extract_features_from_query(parsed,table_stats,columns)
        label = act_rows
        features.append(feature)
        labels.append(max(label,1))
        label_max = max(label_max,label)
    return features, labels


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, queries, table_stats, columns):
        super().__init__()
        self.query_data = list(zip*(preprocess_queries(queries, table_stats, columns)))

    def __getitem__(self, index):
        return self.query_data[index]

    def __len__(self):
        return len(self.query_data)


def est_mlp(train_data, test_data, table_stats, columns):
    """
    est_mlp uses MLP to produce estimated rows for train_data and test_data
    """
    train_dataset = QueryDataset(train_data, table_stats, columns)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
    train_est_rows, train_act_rows = [], []
    # YOUR CODE HERE: train procedure
def weights_init(m):
    classname = m._class_._name_
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight,0.0,0.1)

    net = nn.Sequential(nn.Linear(15,64),nn.ReLU(),nn.Linear(64,1),nn.ReLU())
    loss_fn = nn.MSELoss()
    tranier = torch.optim.Adam(net.parameters(),lr=0.01)
    num_epochs = 30
    net = net.cuda()
    net.apply((weights_init()))
    net.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        total = 0
        for training_features,training_labels in iter(train_loader):
            l = loss_fn(net(training_features.cuda()).reshape(-1),torch.log(training_labels.cuda()))
            tranier.zero_grad()
            l.backward()
            epoch_losses.append((1,training_features.shape[0]))
            total += training_features.shape[0]
        total_loss = torch.tensor(0.,dtype=torch.float32)
        for l, n in epoch_losses :
            total_loss += l.cpu() * n / total
        print("epoch {},loss:{}".format(epoch,l.item()))
    test_dataset = QueryDataset(test_data, table_stats, columns)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_est_rows, test_act_rows = [], []
    # YOUR CODE HERE: test procedure

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def est_xgb(train_data, test_data, table_stats, columns):
    """
    est_xgb uses xgboost to produce estimated rows for train_data and test_data
    """
    print("estimate row counts by xgboost")
    train_x, train_y = preprocess_queries(train_data, table_stats, columns)
    train_est_rows, train_act_rows = [], []
    # YOUR CODE HERE: train procedure
    # train_x = np.array(train_x)
    # train_y = np.array(train_y)
    xgtrain = xgb.DMatrix(train_x, train_y)
    param = {
        'max_depth': 5,
        'n_estimators': 500,
        'learning_rate': 0.1,
        'num_round': 200,
        'eta': 0.1,
        'silent': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        # 'objective': q_error,
        # 'objective': 'reg:squaredlogerror'
    }
    # estimator = xgb.train(param, dtrain=xgtrain)
    estimator = xgb.XGBRegressor(**param)
    estimator.fit(train_x, train_y)
    # train_est_rows = estimator.predict(xgtrain)
    train_est_rows = estimator.predict(train_x)
    test_x, test_y = preprocess_queries(test_data, table_stats, columns)
    test_est_rows, test_act_rows = [], []
    # YOUR CODE HERE: test procedure
    # test_x = np.array(test_x)
    # test_y = np.array(test_y)
    # xgtest = xgb.DMatrix(test_x, test_y)
    # test_est_rows = estimator.predict(xgtest)
    test_est_rows = estimator.predict(test_x)
    train_act_rows = train_y
    test_act_rows = test_y
    test_est_rows = test_est_rows.tolist()
    train_est_rows = train_est_rows.tolist()
    # print(test_est_rows)
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

    eval_model('mlp', train_data, test_data, table_stats, columns)
    eval_model('xgb', train_data, test_data, table_stats, columns)
