from plan import Operator
import torch
import torch.nn as nn


operators = ["Projection", "Selection", "Sort", "HashAgg", "HashJoin", "TableScan", "IndexScan", "TableRowIDScan", "TableReader", "IndexReader", "IndexLookUp"]


class PlanFeatures:
    def __init__(self):
        self.op_count = 0
        # self.op_types = []
        # self.op_est_rows = []
        self.count_per_op = [0] * len(operators)
        self.act_rows_per_op = [0] * len(operators)

    def add_operator(self, op: Operator):
        op_idx = None
        for i, op_name in enumerate(operators):
            if op_name in op.id:
                op_idx = i
                break
            if op_name == "TableScan" and op.is_table_scan():
                op_idx = i
                break
            if op_name == "IndexScan" and op.is_index_scan():
                op_idx = i
                break
        if op_idx is None:
            print(op.id)
        assert op_idx is not None
        self.op_count += 1
        self.count_per_op[op_idx] += 1
        self.act_rows_per_op[op_idx] += float(op.act_rows)

    def walk_operator_tree(self, op: Operator):
        self.add_operator(op)
        for child in op.children:
            self.walk_operator_tree(child)


class DFSOrderFeatures:
    def __init__(self):
        self.vec = []

    def add_operator(self, op: Operator):
        feat = [0.] * len(operators)
        op_idx = None
        for i, op_name in enumerate(operators):
            if op_name in op.id:
                op_idx = i
                break
            if op_name == "TableScan" and op.is_table_scan():
                op_idx = i
                break
            if op_name == "IndexScan" and op.is_index_scan():
                op_idx = i
                break
        assert op_idx is not None
        feat[op_idx] = 1.
        feat.append(float(op.act_rows))
        self.vec.extend(feat)

    def walk_operator_tree(self, op: Operator):
        self.add_operator(op)
        for child in op.children:
            self.walk_operator_tree(child)
        self.vec.extend([0.] * (len(operators) + 1))


class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, plans, max_operator_num):
        super().__init__()
        self.data = []
        for plan in plans:
            # features = PlanFeatures()
            # features.walk_operator_tree(plan.root)
            # features = torch.Tensor([features.op_count] + features.count_per_op + features.act_rows_per_op)
            features = DFSOrderFeatures()
            features.walk_operator_tree(plan.root)
            features.vec.extend([0.] * (max_operator_num * (len(operators) + 1) - len(features.vec)))
            # print(f"len(vec):{len(features.vec)}")
            feats = torch.Tensor(features.vec)
            exec_time = torch.Tensor([plan.exec_time_in_ms()])
            self.data.append((feats, exec_time))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MLP(nn.Module):
    def __init__(self, max_operator_num):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(max_operator_num * (len(operators) + 1), 256),
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




def count_operator_num(op: Operator):
    num = 2 # one for the node and another for None
    for child in op.children:
        num += count_operator_num(child)
    return num


def estimate_learning(train_plans, test_plans):
    max_operator_num = 0
    for plan in train_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    for plan in test_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    print(f"max_operator_num:{max_operator_num}")

    train_dataset = PlanDataset(train_plans, max_operator_num)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=1)

    mlp = MLP(max_operator_num)
    mlp.init_weights()

    def loss_fn(est_time, act_time):
        return torch.mean(torch.abs(est_time - act_time) / act_time)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    num_epoch = 30
    for epoch in range(num_epoch):
        print(f"epoch {epoch} start")
        total_loss = 0.0
        for i, data in enumerate(train_loader):
            features, exec_times = data
            optimizer.zero_grad()
            outputs = mlp(features)
            # print(f"features={features}")
            # print(f"outputs={outputs}")
            # print(f"exec_times={exec_times}")
            loss = loss_fn(outputs, exec_times)
            loss.backward()
            optimizer.step()
            # print(f"i={i}, loss={loss.item()}")
            # if i % 100 == 0:
            #     print(f"i={i}, loss={loss.item()}")
            total_loss += loss.item()
        print(f"epoch {epoch} finish, total_loss={format(total_loss, '.10E')}")

    train_est_times, train_act_times = [], []
    for i, data in enumerate(train_loader):
        features, act_times = data
        est_times = mlp(features)
        # print(f"i={i}, est={est_times}, act={act_times}")
        train_est_times.extend(est_times.tolist())
        train_act_times.extend(act_times.tolist())

    test_dataset = PlanDataset(test_plans, max_operator_num)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)

    test_est_times, test_act_times = [], []
    for i, data in enumerate(test_loader):
        features, act_times = data
        est_times = mlp(features)
        # print(f"i={i}, est={est_times}, act={act_times}")
        test_est_times.extend(est_times.tolist())
        test_act_times.extend(act_times.tolist())

    return train_est_times, train_act_times, test_est_times, test_act_times
