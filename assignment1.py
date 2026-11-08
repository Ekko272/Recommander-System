import bisect
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix


# Xuanyu Liu A18502378 Hw 3 & assignment 1
# AI usage Claimer: I used ChatGPT to help me understand the code and to debug the code. 
# I also used the code from the lecture notes.


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NEIGH_TOPK = 15
NEIGH_SHRINK = 1.0
NEIGH_CO = 3.0
REG_USER = 1.0
REG_ITEM = 5.0

BPR_FACTORS = 64
BPR_LR = 0.05
BPR_REG = 0.01
BPR_EPOCHS = 12
BPR_BATCH = 4096
BPR_NEGATIVES = 4

SEQ_DIM = 96
SEQ_WINDOW = 12
SEQ_EPOCHS = 8
SEQ_BATCH = 8192
SEQ_NEGATIVES = 12
SEQ_LR = 0.002
SEQ_COS_DECAY = 0.7

GAP_PRIOR = 40.0
GAP_EPOCHS = 12
GAP_LR = 0.05
GAP_BATCH = 8192
DAY_SECONDS = 86400.0

def load_train(path):
    users_s, items_s, times, ratings = [], [], [], []
    with open(path) as handle:
        handle.readline()
        for line in handle:
            user, item, timestamp, rating = line.strip().split(",")
            users_s.append(user)
            items_s.append(item)
            times.append(int(timestamp))
            ratings.append(float(rating))
    user_map, item_map = {}, {}
    user_ids = np.empty(len(users_s), np.int64)
    item_ids = np.empty(len(items_s), np.int64)
    for index, user in enumerate(users_s):
        user_ids[index] = user_map.setdefault(user, len(user_map))
    for index, item in enumerate(items_s):
        item_ids[index] = item_map.setdefault(item, len(item_map))
    return (
        user_ids,
        item_ids,
        np.asarray(times, np.int64),
        np.asarray(ratings, np.float64),
        user_map,
        item_map,
    )


def fit_biases(users, items, ratings, n_users, n_items, reg_user=REG_USER, reg_item=REG_ITEM, iters=10):
    mu = float(ratings.mean())
    user_count = np.bincount(users, minlength=n_users).astype(np.float64)
    item_count = np.bincount(items, minlength=n_items).astype(np.float64)
    user_bias = np.zeros(n_users, np.float64)
    item_bias = np.zeros(n_items, np.float64)
    for _ in range(iters):
        residual = ratings - mu - item_bias[items]
        user_bias = np.bincount(users, weights=residual, minlength=n_users) / (user_count + reg_user)
        residual = ratings - mu - user_bias[users]
        item_bias = np.bincount(items, weights=residual, minlength=n_items) / (item_count + reg_item)
    return mu, user_bias, item_bias


def group_bounds(ids):
    cuts = np.flatnonzero(np.diff(ids))
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts + 1, len(ids)]
    return starts, ends


def build_similarity(users, items, n_users, n_items):
    values = np.ones(len(users), np.float64)
    matrix = csr_matrix((values, (items, users)), shape=(n_items, n_users))
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1))).ravel() + 1e-8
    normalized = matrix.multiply((1.0 / norms)[:, None]).tocsr()
    return matrix, normalized


def build_histories(n_users, users, items, ratings, times):
    histories = [[] for _ in range(n_users)]
    for user, item, rating, timestamp in zip(
        users.tolist(), items.tolist(), ratings.tolist(), times.tolist()
    ):
        histories[user].append((item, rating, timestamp))
    for events in histories:
        events.sort(key=lambda event: event[2])
    return histories


class RatingModel:
    def __init__(self, mu, user_bias, item_bias, counts, cosine, histories, topk, shrink, co_shrink):
        self.mu = mu
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.counts = counts
        self.cosine = cosine
        self.histories = histories
        self.topk = topk
        self.shrink = shrink
        self.co_shrink = co_shrink
        self.tau_days = None
        self.bin_bias = None
        self.n_bins = 24
        self.time_lo = 1.15e9
        self.time_hi = 1.52e9

    def time_bin(self, timestamp):
        clipped = min(self.time_hi, max(self.time_lo, float(timestamp)))
        bin_id = int((clipped - self.time_lo) / (self.time_hi - self.time_lo) * self.n_bins)
        return min(self.n_bins - 1, max(0, bin_id))

    def predict(self, user, item, timestamp=None):
        mu = self.mu
        base = mu + self.user_bias[user] + self.item_bias[item]
        events = self.histories[user]
        if events:
            item_ids = np.fromiter((event[0] for event in events), np.int64, len(events))
            ratings = np.fromiter((event[1] for event in events), np.float64, len(events))
            sims = np.asarray(self.cosine[item_ids].dot(self.cosine[item].T).todense()).ravel()
            common = np.asarray(self.counts[item_ids].dot(self.counts[item].T).todense()).ravel()
            sims = sims * common / (common + self.co_shrink)
            if self.tau_days is not None and timestamp is not None:
                when = np.fromiter((event[2] for event in events), np.float64, len(events))
                sims = sims * np.exp(-np.abs(when - float(timestamp)) / (self.tau_days * 86400.0))
            if len(sims) > self.topk:
                keep = np.argpartition(sims, -self.topk)[-self.topk :]
                sims = sims[keep]
                ratings = ratings[keep]
                item_ids = item_ids[keep]
            positive = sims > 0
            if np.any(positive):
                sims = sims[positive]
                ratings = ratings[positive]
                item_ids = item_ids[positive]
                neighbor_base = mu + self.user_bias[user] + self.item_bias[item_ids]
                correction = float(np.dot(sims, ratings - neighbor_base)) / (float(sims.sum()) + self.shrink)
                base = base + correction
        if self.bin_bias is not None and timestamp is not None:
            base = base + self.bin_bias[self.time_bin(timestamp)]
        return float(min(5.0, max(0.0, base)))


def make_rating_model(users, items, ratings, times, n_users, n_items):
    mu, user_bias, item_bias = fit_biases(users, items, ratings, n_users, n_items)
    counts, cosine = build_similarity(users, items, n_users, n_items)
    histories = build_histories(n_users, users, items, ratings, times)
    return RatingModel(
        mu, user_bias, item_bias, counts, cosine, histories, NEIGH_TOPK, NEIGH_SHRINK, NEIGH_CO
    )


def mean_squared_error(model, users, items, ratings, times, indices, use_time=False):
    error = 0.0
    for index in indices:
        timestamp = int(times[index]) if use_time else None
        prediction = model.predict(int(users[index]), int(items[index]), timestamp)
        error += (prediction - float(ratings[index])) ** 2
    return error / len(indices)


def fit_bin_bias(model, users, items, ratings, times, indices):
    residual_sum = np.zeros(model.n_bins, np.float64)
    residual_count = np.zeros(model.n_bins, np.float64)
    for index in indices:
        user = int(users[index])
        item = int(items[index])
        timestamp = int(times[index])
        base = model.mu + model.user_bias[user] + model.item_bias[item]
        bin_id = model.time_bin(timestamp)
        residual_sum[bin_id] += float(ratings[index]) - base
        residual_count[bin_id] += 1.0
    return residual_sum / (residual_count + 150.0)


class BPR(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.item_bias = nn.Embedding(n_items, 1)
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.user_factors.weight, std=0.1)
        nn.init.normal_(self.item_factors.weight, std=0.1)

    def score(self, users, items):
        return self.item_bias(items).squeeze(-1) + (self.user_factors(users) * self.item_factors(items)).sum(-1)


def train_bpr(users, items, n_users, n_items, seed=0):
    torch.manual_seed(seed)
    model = BPR(n_users, n_items, BPR_FACTORS).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=BPR_LR)
    user_tensor = torch.as_tensor(users)
    item_tensor = torch.as_tensor(items)
    popularity = torch.bincount(item_tensor, minlength=n_items).float()
    popularity = popularity / popularity.sum()
    popularity = popularity.to(DEVICE)
    n_rows = user_tensor.shape[0]
    for epoch in range(BPR_EPOCHS):
        model.train()
        order = torch.randperm(n_rows)
        for start in range(0, n_rows, BPR_BATCH):
            batch = order[start : start + BPR_BATCH]
            batch_users = user_tensor[batch].to(DEVICE)
            batch_items = item_tensor[batch].to(DEVICE)
            batch_size = batch_users.shape[0]
            negatives = torch.multinomial(
                popularity, batch_size * BPR_NEGATIVES, replacement=True
            ).view(batch_size, BPR_NEGATIVES)
            positive = model.score(batch_users, batch_items)
            negative = model.score(batch_users[:, None].expand(batch_size, BPR_NEGATIVES), negatives)
            loss = -F.logsigmoid(positive[:, None] - negative).sum()
            loss = loss + 0.5 * BPR_REG * (
                model.user_factors(batch_users).pow(2).sum()
                + model.item_factors(batch_items).pow(2).sum()
                + model.item_factors(negatives).pow(2).sum()
                + model.item_bias(batch_items).pow(2).sum()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"    bpr epoch {epoch + 1}/{BPR_EPOCHS}", flush=True)
    model.eval()
    return model


def bpr_score_pairs(model, users, items):
    scores = np.empty(len(users), np.float64)
    with torch.no_grad():
        for start in range(0, len(users), 65536):
            user_batch = torch.as_tensor(users[start : start + 65536], device=DEVICE)
            item_batch = torch.as_tensor(items[start : start + 65536], device=DEVICE)
            scores[start : start + 65536] = model.score(user_batch, item_batch).float().cpu().numpy()
    return scores


def cosine_affinity(cosine, history_items, candidate, topk):
    if len(history_items) == 0:
        return 0.0
    sims = np.asarray(cosine[history_items].dot(cosine[candidate].T).todense()).ravel()
    if len(sims) > topk:
        sims = np.partition(sims, -topk)[-topk:]
    return float(np.clip(sims, 0, None).mean())


def nearest_items(events, timestamp, k):
    if not events:
        return np.empty(0, np.int64)
    times = [event[2] for event in events]
    cursor = bisect.bisect_left(times, timestamp)
    left, right = cursor, cursor
    chosen = []
    while len(chosen) < k and (left > 0 or right < len(events)):
        left_gap = abs(times[left - 1] - timestamp) if left > 0 else None
        right_gap = abs(times[right] - timestamp) if right < len(events) else None
        if right_gap is None or (left_gap is not None and left_gap <= right_gap):
            left -= 1
            chosen.append(events[left][0])
        else:
            chosen.append(events[right][0])
            right += 1
    return np.asarray(chosen, np.int64)


def read_test_rows(path):
    with open(path) as handle:
        header = handle.readline()
        rows = []
        for line in handle:
            line = line.strip()
            if line:
                rows.append(line.split(","))
    if not header.endswith("\n"):
        header += "\n"
    return header, rows


def write_predictions(path, header, lines):
    with open(path, "w") as handle:
        handle.write(header)
        for line in lines:
            handle.write(line + "\n")


def holdout_split(n_rows, fraction=0.05, seed=0):
    generator = np.random.default_rng(seed)
    order = generator.permutation(n_rows)
    n_val = int(fraction * n_rows)
    return order[:n_val], order[n_val:]


def task1():
    print("Task 1: rating prediction", flush=True)
    started = time.time()
    users, items, times, ratings, user_map, item_map = load_train(
        os.path.join(DATA_DIR, "tasks_1_2_train.csv")
    )
    n_users, n_items = len(user_map), len(item_map)
    val_index, train_index = holdout_split(len(users), seed=0)
    # Confirm the neighborhood on a slice that was not used to pick its settings.
    check = val_index[5000:10000]
    model = make_rating_model(
        users[train_index], items[train_index], ratings[train_index], times[train_index], n_users, n_items
    )
    heldout_mse = mean_squared_error(model, users, items, ratings, times, check, use_time=False)
    print(f"  held-out MSE {heldout_mse:.4f}", flush=True)

    full = make_rating_model(users, items, ratings, times, n_users, n_items)
    header, rows = read_test_rows(os.path.join(DATA_DIR, "task_1_test.csv"))
    lines = []
    for user_str, item_str in rows:
        user = user_map[user_str]
        item = item_map[item_str]
        lines.append(f"{user_str},{item_str},{full.predict(user, item):.6f}")
    write_predictions(os.path.join(DATA_DIR, "task_1_predictions.csv"), header, lines)
    print(f"  wrote {len(lines)} predictions in {time.time() - started:.1f}s", flush=True)
    return heldout_mse


def tune_blend(scores):
    positive = scores[:, 0, :]
    negative = scores[:, 1, :]
    n_extra = positive.shape[1] - 1
    if n_extra == 1:
        best_weight, best_acc = 0.0, -1.0
        for weight in np.linspace(0.0, 8.0, 33):
            accuracy = float(np.mean(positive[:, 0] + weight * positive[:, 1] >= negative[:, 0] + weight * negative[:, 1]))
            if accuracy > best_acc:
                best_acc, best_weight = accuracy, float(weight)
        print(f"  blend weight {best_weight:.2f}  holdout accuracy {best_acc:.4f}", flush=True)
        return [best_weight], best_acc
    best = (0.0, 0.0)
    best_acc = -1.0
    for weight_context in np.linspace(0.0, 24.0, 25):
        for weight_global in np.linspace(0.0, 12.0, 13):
            score_pos = positive[:, 0] + weight_context * positive[:, 1] + weight_global * positive[:, 2]
            score_neg = negative[:, 0] + weight_context * negative[:, 1] + weight_global * negative[:, 2]
            accuracy = float(np.mean(score_pos >= score_neg))
            if accuracy > best_acc:
                best_acc = accuracy
                best = (float(weight_context), float(weight_global))
    print(
        f"  blend weights context={best[0]:.2f} global={best[1]:.2f}  holdout accuracy {best_acc:.4f}",
        flush=True,
    )
    return list(best), best_acc


def ranking_features(cosine, histories, model, users, item_a, item_b, times=None, context_k=10, global_k=20):
    bpr_a = bpr_score_pairs(model, users, item_a)
    bpr_b = bpr_score_pairs(model, users, item_b)
    extra_a = []
    extra_b = []
    for row, user in enumerate(users):
        events = histories[int(user)]
        if times is None:
            history_items = np.fromiter((event[0] for event in events), np.int64, len(events)) if events else np.empty(0, np.int64)
            extra_a.append([cosine_affinity(cosine, history_items, int(item_a[row]), global_k)])
            extra_b.append([cosine_affinity(cosine, history_items, int(item_b[row]), global_k)])
        else:
            context = nearest_items(events, int(times[row]), context_k)
            history_items = np.fromiter((event[0] for event in events), np.int64, len(events)) if events else np.empty(0, np.int64)
            extra_a.append(
                [
                    cosine_affinity(cosine, context, int(item_a[row]), context_k),
                    cosine_affinity(cosine, history_items, int(item_a[row]), global_k),
                ]
            )
            extra_b.append(
                [
                    cosine_affinity(cosine, context, int(item_b[row]), context_k),
                    cosine_affinity(cosine, history_items, int(item_b[row]), global_k),
                ]
            )
    features_a = np.column_stack([bpr_a, np.asarray(extra_a, np.float64)])
    features_b = np.column_stack([bpr_b, np.asarray(extra_b, np.float64)])
    return np.stack([features_a, features_b], axis=1)


def sample_ranking_questions(users, items, times, train_index, val_index, n_questions, seed):
    generator = np.random.default_rng(seed)
    train_items = items[train_index]
    chosen = val_index if len(val_index) <= n_questions else generator.choice(val_index, n_questions, replace=False)
    seen = [set() for _ in range(int(users.max()) + 1)]
    for user, item in zip(users[train_index].tolist(), items[train_index].tolist()):
        seen[user].add(item)
    question_users = users[chosen]
    positives = items[chosen]
    negatives = np.empty(len(chosen), np.int64)
    question_times = times[chosen] if times is not None else None
    for row, (user, positive) in enumerate(zip(question_users, positives)):
        history = seen[int(user)]
        for _ in range(40):
            candidate = int(train_items[generator.integers(0, len(train_items))])
            if candidate != int(positive) and candidate not in history:
                negatives[row] = candidate
                break
        else:
            negatives[row] = int(positive)
    return question_users, positives, negatives, question_times


def task2():
    print("Task 2: interaction prediction", flush=True)
    started = time.time()
    users, items, times, ratings, user_map, item_map = load_train(
        os.path.join(DATA_DIR, "tasks_1_2_train.csv")
    )
    n_users, n_items = len(user_map), len(item_map)
    val_index, train_index = holdout_split(len(users), seed=1)
    print("  training blend model", flush=True)
    blend_model = train_bpr(users[train_index], items[train_index], n_users, n_items, seed=1)
    _, cosine = build_similarity(users[train_index], items[train_index], n_users, n_items)
    histories = build_histories(
        n_users, users[train_index], items[train_index], ratings[train_index], times[train_index]
    )
    q_users, q_pos, q_neg, _ = sample_ranking_questions(
        users, items, None, train_index, val_index, n_questions=6000, seed=2
    )
    scores = ranking_features(cosine, histories, blend_model, q_users, q_pos, q_neg, times=None)
    weights, accuracy = tune_blend(scores)

    print("  retraining on all interactions", flush=True)
    model = train_bpr(users, items, n_users, n_items, seed=1)
    _, cosine = build_similarity(users, items, n_users, n_items)
    histories = build_histories(n_users, users, items, ratings, times)
    header, rows = read_test_rows(os.path.join(DATA_DIR, "task_2_test.csv"))
    test_users = np.array([user_map[row[0]] for row in rows], np.int64)
    item1 = np.array([item_map[row[1]] for row in rows], np.int64)
    item2 = np.array([item_map[row[2]] for row in rows], np.int64)
    # Score in chunks so the similarity pass can print progress on a long test file.
    lines = []
    weight = weights[0]
    bpr1 = bpr_score_pairs(model, test_users, item1)
    bpr2 = bpr_score_pairs(model, test_users, item2)
    for row_index, row in enumerate(rows):
        events = histories[int(test_users[row_index])]
        history_items = (
            np.fromiter((event[0] for event in events), np.int64, len(events))
            if events
            else np.empty(0, np.int64)
        )
        sim1 = cosine_affinity(cosine, history_items, int(item1[row_index]), 20)
        sim2 = cosine_affinity(cosine, history_items, int(item2[row_index]), 20)
        choose_first = bpr1[row_index] + weight * sim1 >= bpr2[row_index] + weight * sim2
        chosen = row[1] if choose_first else row[2]
        lines.append(f"{row[0]},{row[1]},{row[2]},{chosen}")
    write_predictions(os.path.join(DATA_DIR, "task_2_predictions.csv"), header, lines)
    print(f"  wrote {len(lines)} predictions in {time.time() - started:.1f}s", flush=True)
    return accuracy


def task3():
    print("Task 3: rating prediction with time", flush=True)
    started = time.time()
    users, items, times, ratings, user_map, item_map = load_train(
        os.path.join(DATA_DIR, "tasks_3_4_train.csv")
    )
    n_users, n_items = len(user_map), len(item_map)
    val_index, train_index = holdout_split(len(users), seed=0)
    check = val_index[:4000]
    model = make_rating_model(
        users[train_index], items[train_index], ratings[train_index], times[train_index], n_users, n_items
    )
    best_tau, best_mse = None, mean_squared_error(model, users, items, ratings, times, check, use_time=True)
    print(f"  tau=none MSE {best_mse:.4f}", flush=True)
    for tau in (30, 90, 180, 365):
        model.tau_days = tau
        mse = mean_squared_error(model, users, items, ratings, times, check, use_time=True)
        print(f"  tau={tau} MSE {mse:.4f}", flush=True)
        if mse < best_mse:
            best_tau, best_mse = tau, mse
    model.tau_days = best_tau
    bin_bias = fit_bin_bias(model, users, items, ratings, times, train_index)
    model.bin_bias = bin_bias
    mse_with_bins = mean_squared_error(model, users, items, ratings, times, check, use_time=True)
    print(f"  with time-bin bias MSE {mse_with_bins:.4f}", flush=True)
    use_bins = mse_with_bins < best_mse
    if use_bins:
        best_mse = mse_with_bins
    print(f"  using tau={best_tau} bins={use_bins}", flush=True)

    full = make_rating_model(users, items, ratings, times, n_users, n_items)
    full.tau_days = best_tau
    if use_bins:
        full.bin_bias = fit_bin_bias(full, users, items, ratings, times, np.arange(len(users)))
    header, rows = read_test_rows(os.path.join(DATA_DIR, "task_3_test.csv"))
    lines = []
    for user_str, item_str, timestamp_str in rows:
        prediction = full.predict(user_map[user_str], item_map[item_str], int(timestamp_str))
        lines.append(f"{user_str},{item_str},{timestamp_str},{prediction:.6f}")
    write_predictions(os.path.join(DATA_DIR, "task_3_predictions.csv"), header, lines)
    print(f"  wrote {len(lines)} predictions in {time.time() - started:.1f}s", flush=True)
    return best_mse


def task4():
    print("Task 4: interaction prediction with time", flush=True)
    started = time.time()
    users, items, times, ratings, user_map, item_map = load_train(
        os.path.join(DATA_DIR, "tasks_3_4_train.csv")
    )
    n_users, n_items = len(user_map), len(item_map)
    val_index, train_index = holdout_split(len(users), seed=1)
    print("  training blend model", flush=True)
    blend_model = train_bpr(users[train_index], items[train_index], n_users, n_items, seed=1)
    _, cosine = build_similarity(users[train_index], items[train_index], n_users, n_items)
    histories = build_histories(
        n_users, users[train_index], items[train_index], ratings[train_index], times[train_index]
    )
    q_users, q_pos, q_neg, q_times = sample_ranking_questions(
        users, items, times, train_index, val_index, n_questions=5000, seed=3
    )
    scores = ranking_features(
        cosine, histories, blend_model, q_users, q_pos, q_neg, times=q_times
    )
    weights, accuracy = tune_blend(scores)

    print("  retraining on all interactions", flush=True)
    model = train_bpr(users, items, n_users, n_items, seed=1)
    _, cosine = build_similarity(users, items, n_users, n_items)
    histories = build_histories(n_users, users, items, ratings, times)
    header, rows = read_test_rows(os.path.join(DATA_DIR, "task_4_test.csv"))
    test_users = np.array([user_map[row[0]] for row in rows], np.int64)
    test_times = np.array([int(row[1]) for row in rows], np.int64)
    item1 = np.array([item_map[row[2]] for row in rows], np.int64)
    item2 = np.array([item_map[row[3]] for row in rows], np.int64)
    bpr1 = bpr_score_pairs(model, test_users, item1)
    bpr2 = bpr_score_pairs(model, test_users, item2)
    weight_context, weight_global = weights
    lines = []
    for row_index, row in enumerate(rows):
        events = histories[int(test_users[row_index])]
        context = nearest_items(events, int(test_times[row_index]), 10)
        history_items = (
            np.fromiter((event[0] for event in events), np.int64, len(events))
            if events
            else np.empty(0, np.int64)
        )
        score1 = (
            bpr1[row_index]
            + weight_context * cosine_affinity(cosine, context, int(item1[row_index]), 10)
            + weight_global * cosine_affinity(cosine, history_items, int(item1[row_index]), 20)
        )
        score2 = (
            bpr2[row_index]
            + weight_context * cosine_affinity(cosine, context, int(item2[row_index]), 10)
            + weight_global * cosine_affinity(cosine, history_items, int(item2[row_index]), 20)
        )
        chosen = row[2] if score1 >= score2 else row[3]
        lines.append(f"{row[0]},{row[1]},{row[2]},{row[3]},{chosen}")
    write_predictions(os.path.join(DATA_DIR, "task_4_predictions.csv"), header, lines)
    print(f"  wrote {len(lines)} predictions in {time.time() - started:.1f}s", flush=True)
    return accuracy


class SeqRanker(nn.Module):
    # Next-item score: user vector plus an attention pool over the recent books.
    def __init__(self, n_users, n_items, n_factors, window):
        super().__init__()
        self.pad = n_items
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.context_factors = nn.Embedding(n_items + 1, n_factors, padding_idx=n_items)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.item_bias = nn.Embedding(n_items, 1)
        self.position = nn.Embedding(window, n_factors)
        self.attention = nn.Linear(n_factors, 1)
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.context_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)
        nn.init.normal_(self.position.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.context_factors.weight[n_items].zero_()

    def query(self, users, context):
        positions = torch.arange(context.shape[1], device=context.device)
        embedded = self.context_factors(context) + self.position(positions)[None, :, :]
        logits = self.attention(embedded).squeeze(-1)
        logits = logits.masked_fill(context == self.pad, -1e9)
        weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        pooled = (embedded * weights[..., None]).sum(1)
        return self.user_factors(users) + pooled

    def score(self, queries, items):
        return self.item_bias(items).squeeze(-1) + (queries * self.item_factors(items)).sum(-1)


def build_next_examples(items, starts, ends, n_items, window, holdout_last):
    lengths = ends - starts
    drop = 2 if holdout_last else 1
    n_examples = int(np.maximum(lengths - drop, 0).sum())
    context = np.full((n_examples, window), n_items, np.int64)
    positives = np.empty(n_examples, np.int64)
    example_users = np.empty(n_examples, np.int64)
    row = 0
    for user, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        sequence = items[start:end]
        last_target = len(sequence) - (1 if holdout_last else 0)
        for cursor in range(1, last_target):
            chunk = sequence[max(0, cursor - window) : cursor]
            context[row, window - len(chunk) :] = chunk
            positives[row] = sequence[cursor]
            example_users[row] = user
            row += 1
    return example_users, context, positives


def train_seq_ranker(n_users, n_items, example_users, context, positives, popularity, epochs=SEQ_EPOCHS):
    torch.manual_seed(0)
    model = SeqRanker(n_users, n_items, SEQ_DIM, context.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=SEQ_LR)
    users_t = torch.as_tensor(example_users, dtype=torch.long)
    context_t = torch.as_tensor(context, dtype=torch.long)
    positives_t = torch.as_tensor(positives, dtype=torch.long)
    popularity_t = torch.as_tensor(popularity, dtype=torch.float32, device=DEVICE)
    n_rows = users_t.shape[0]
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n_rows)
        for start in range(0, n_rows, SEQ_BATCH):
            batch = order[start : start + SEQ_BATCH]
            batch_users = users_t[batch].to(DEVICE)
            batch_context = context_t[batch].to(DEVICE)
            batch_items = positives_t[batch].to(DEVICE)
            batch_size = batch_users.shape[0]
            negatives = torch.multinomial(popularity_t, batch_size * SEQ_NEGATIVES, replacement=True)
            negatives = negatives.view(batch_size, SEQ_NEGATIVES)
            queries = model.query(batch_users, batch_context)
            positive = model.score(queries, batch_items)
            expanded = queries[:, None, :].expand(batch_size, SEQ_NEGATIVES, SEQ_DIM).reshape(-1, SEQ_DIM)
            negative = model.score(expanded, negatives.reshape(-1)).view(batch_size, SEQ_NEGATIVES)
            loss = -F.logsigmoid(positive[:, None] - negative).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"    seq epoch {epoch + 1}/{epochs}", flush=True)
    model.eval()
    return model


def score_seq_pairs(model, users, context, item_a, item_b):
    scores_a = np.zeros(len(users), np.float64)
    scores_b = np.zeros(len(users), np.float64)
    with torch.no_grad():
        queries = model.query(
            torch.as_tensor(users, dtype=torch.long, device=DEVICE),
            torch.as_tensor(context, dtype=torch.long, device=DEVICE),
        )
        for scores, items in ((scores_a, item_a), (scores_b, item_b)):
            known = items >= 0
            if not np.any(known):
                continue
            index = torch.as_tensor(np.flatnonzero(known), dtype=torch.long, device=DEVICE)
            item_tensor = torch.as_tensor(items[known], dtype=torch.long, device=DEVICE)
            scores[known] = model.score(queries[index], item_tensor).float().cpu().numpy()
    return scores_a, scores_b


def pairwise_cosine(cosine, left, right):
    scores = np.empty(len(left), np.float64)
    for start in range(0, len(left), 8000):
        stop = start + 8000
        block = cosine[left[start:stop]].multiply(cosine[right[start:stop]]).sum(axis=1)
        scores[start:stop] = np.asarray(block).ravel()
    return scores


def recency_cosine(cosine, history, candidates, decay=SEQ_COS_DECAY):
    # candidates is a list of item ids; unknown ids contribute nothing.
    scores = np.zeros(len(candidates), np.float64)
    if len(history) == 0:
        return scores
    weights = decay ** np.arange(len(history) - 1, -1, -1)
    weight_sum = float(weights.sum())
    for index, candidate in enumerate(candidates):
        if candidate < 0:
            continue
        sims = np.asarray(cosine[history].dot(cosine[candidate].T).todense()).ravel()
        scores[index] = float(np.dot(weights, np.clip(sims, 0, None)) / weight_sum)
    return scores


def right_context(sequence, window, pad):
    context = np.full(window, pad, np.int64)
    chunk = sequence[-window:]
    if len(chunk):
        context[window - len(chunk) :] = chunk
    return context


def task5():
    print("Task 5: next-item prediction", flush=True)
    started = time.time()
    users, items, times, ratings, user_map, item_map = load_train(os.path.join(DATA_DIR, "task_5_train.csv"))
    order = np.lexsort((times, users))
    users, items = users[order], items[order]
    n_users, n_items = len(user_map), len(item_map)
    starts, ends = group_bounds(users)
    if not np.array_equal(users[starts], np.arange(n_users)):
        raise RuntimeError("user ids are not contiguous after sorting")

    popularity = np.bincount(items, minlength=n_items).astype(np.float64)
    popularity = popularity / popularity.sum()
    print("  training holdout ranker", flush=True)
    holdout_users, holdout_context, holdout_items = build_next_examples(
        items, starts, ends, n_items, SEQ_WINDOW, holdout_last=True
    )
    holdout_model = train_seq_ranker(n_users, n_items, holdout_users, holdout_context, holdout_items, popularity)

    generator = np.random.default_rng(0)
    long_enough = np.flatnonzero((ends - starts) >= 4)
    sample = generator.choice(long_enough, size=min(8000, len(long_enough)), replace=False)
    prefix_keep = np.ones(len(users), np.bool_)
    prefix_keep[ends - 1] = False
    _, prefix_cosine = build_similarity(users[prefix_keep], items[prefix_keep], n_users, n_items)
    sample_context = np.full((len(sample), SEQ_WINDOW), n_items, np.int64)
    sample_positive = np.empty(len(sample), np.int64)
    sample_negative = np.empty(len(sample), np.int64)
    positive_cosine = np.zeros(len(sample), np.float64)
    negative_cosine = np.zeros(len(sample), np.float64)
    catalog = items
    for row, user in enumerate(sample.tolist()):
        sequence = items[int(starts[user]) : int(ends[user])]
        history = sequence[:-1]
        positive = int(sequence[-1])
        seen = set(history.tolist())
        negative = positive
        for _ in range(40):
            candidate = int(catalog[generator.integers(0, len(catalog))])
            if candidate != positive and candidate not in seen:
                negative = candidate
                break
        sample_positive[row] = positive
        sample_negative[row] = negative
        sample_context[row] = right_context(history, SEQ_WINDOW, n_items)
        recent = history[-SEQ_WINDOW:]
        positive_cosine[row], negative_cosine[row] = recency_cosine(
            prefix_cosine, recent, [positive, negative]
        )
    model_positive, model_negative = score_seq_pairs(
        holdout_model, sample.astype(np.int64), sample_context, sample_positive, sample_negative
    )
    model_gap = model_positive - model_negative
    model_accuracy = float(np.mean(model_gap > 0) + 0.5 * np.mean(model_gap == 0))
    best_weight, best_accuracy = 0.0, model_accuracy
    cosine_gap = positive_cosine - negative_cosine
    for weight in np.linspace(0.0, 40.0, 21):
        gap = model_gap + weight * cosine_gap
        accuracy = float(np.mean(gap > 0) + 0.5 * np.mean(gap == 0))
        if accuracy > best_accuracy:
            best_accuracy, best_weight = accuracy, float(weight)
    print(
        f"  holdout model {model_accuracy:.4f}  cosine weight {best_weight:.1f}  "
        f"blended {best_accuracy:.4f}",
        flush=True,
    )

    print("  retraining on all transitions", flush=True)
    full_users, full_context, full_items = build_next_examples(
        items, starts, ends, n_items, SEQ_WINDOW, holdout_last=False
    )
    model = train_seq_ranker(n_users, n_items, full_users, full_context, full_items, popularity)
    _, cosine = build_similarity(users, items, n_users, n_items)
    header, rows = read_test_rows(os.path.join(DATA_DIR, "task_5_test.csv"))
    test_users = np.array([user_map[row[0]] for row in rows], np.int64)
    item1 = np.array([item_map.get(row[1], -1) for row in rows], np.int64)
    item2 = np.array([item_map.get(row[2], -1) for row in rows], np.int64)
    test_context = np.full((len(rows), SEQ_WINDOW), n_items, np.int64)
    for row, user in enumerate(test_users.tolist()):
        test_context[row] = right_context(items[int(starts[user]) : int(ends[user])], SEQ_WINDOW, n_items)
    score1, score2 = score_seq_pairs(model, test_users, test_context, item1, item2)
    if best_weight > 0:
        for row, user in enumerate(test_users.tolist()):
            history = items[int(starts[user]) : int(ends[user])][-SEQ_WINDOW:]
            extra1, extra2 = recency_cosine(cosine, history, [int(item1[row]), int(item2[row])])
            score1[row] += best_weight * extra1
            score2[row] += best_weight * extra2
    lines = []
    for row, (user_str, left, right) in enumerate(rows):
        chosen = left if score1[row] >= score2[row] else right
        lines.append(f"{user_str},{left},{right},{chosen}")
    write_predictions(os.path.join(DATA_DIR, "task_5_predictions.csv"), header, lines)
    print(f"  wrote {len(lines)} predictions in {time.time() - started:.1f}s", flush=True)
    return best_accuracy


def item_gap_prior(destinations, gaps, n_items, prior):
    global_median = float(np.median(gaps))
    order = np.argsort(destinations, kind="mergesort")
    ordered_items = destinations[order]
    ordered_gaps = gaps[order]
    change = np.flatnonzero(np.diff(ordered_items))
    bounds = np.r_[0, change + 1, len(ordered_items)]
    median = np.zeros(n_items, np.float64)
    count = np.zeros(n_items, np.float64)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        item = int(ordered_items[start])
        count[item] = stop - start
        median[item] = float(np.median(ordered_gaps[start:stop]))
    shrunk = (median * count + prior * global_median) / (count + prior)
    return shrunk, global_median


def consecutive_edges(users, items, times):
    same_user = users[1:] == users[:-1]
    return items[:-1][same_user], items[1:][same_user], (times[1:] - times[:-1])[same_user].astype(np.float64)


def fit_gap_model(features, target):
    center = features.mean(axis=0)
    scale = features.std(axis=0) + 1e-6
    standardized = (features - center) / scale
    target_t = torch.as_tensor(np.log1p(target), dtype=torch.float32, device=DEVICE)
    design = torch.as_tensor(standardized, dtype=torch.float32, device=DEVICE)
    weight = torch.zeros(design.shape[1], device=DEVICE, requires_grad=True)
    bias = torch.tensor(float(target_t.mean()), device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=GAP_LR)
    n_rows = design.shape[0]
    for _ in range(GAP_EPOCHS):
        permutation = torch.randperm(n_rows, device=DEVICE)
        for start in range(0, n_rows, GAP_BATCH):
            batch = permutation[start : start + GAP_BATCH]
            residual = target_t[batch] - (design[batch] @ weight + bias)
            loss = residual.abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return weight.detach().cpu().numpy(), float(bias.detach().cpu()), center, scale


def predict_gap(features, weight, bias, center, scale):
    standardized = (features - center) / scale
    raw = np.expm1(standardized @ weight + bias)
    return np.clip(raw, 60.0, 5.0e8)


def gap_feature_row(previous_gaps, previous_log, timestamp, rating, item_typical, popularity):
    median = float(np.median(previous_gaps))
    geometric = float(np.expm1(previous_log.mean()))
    last = float(previous_gaps[-1])
    recent = float(np.median(previous_gaps[-min(3, len(previous_gaps)) :]))
    hour = (int(timestamp) % 86400) / 3600.0
    weekday = (int(timestamp) // 86400) % 7
    return [
        np.log1p(median),
        np.log1p(geometric),
        np.log1p(last),
        np.log1p(recent),
        np.log1p(last) - np.log1p(median),
        np.log1p(len(previous_gaps)),
        float(rating),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * weekday / 7.0),
        np.cos(2 * np.pi * weekday / 7.0),
        float(np.mean(previous_gaps < DAY_SECONDS)),
        np.log1p(item_typical),
        np.log1p(popularity),
    ]


def collect_gap_examples(starts, ends, items, times, ratings, item_typical, popularity, include_last):
    feature_rows = []
    labels = []
    previous_items = []
    destination_items = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        if end - start < 4:
            continue
        stamps = times[start:end]
        sequence = items[start:end]
        rate = ratings[start:end]
        deltas = np.diff(stamps.astype(np.float64))
        logged = np.log1p(deltas)
        last_index = len(deltas) if include_last else len(deltas) - 1
        for index in range(1, last_index):
            if len(deltas) > 12 and index < len(deltas) - 7 and index % 3 != 0:
                continue
            feature_rows.append(
                gap_feature_row(
                    deltas[:index],
                    logged[:index],
                    stamps[index],
                    rate[index],
                    item_typical[int(sequence[index + 1])],
                    popularity[int(sequence[index + 1])],
                )
            )
            labels.append(float(deltas[index]))
            previous_items.append(int(sequence[index]))
            destination_items.append(int(sequence[index + 1]))
    return (
        np.asarray(feature_rows, np.float64),
        np.asarray(labels, np.float64),
        np.asarray(previous_items, np.int64),
        np.asarray(destination_items, np.int64),
    )


def attach_book_features(features, cosine, transitions, previous_items, destination_items):
    similarity = pairwise_cosine(cosine, previous_items, destination_items)
    counts = np.fromiter(
        (transitions.get((int(left), int(right)), 0) for left, right in zip(previous_items, destination_items)),
        np.float64,
        count=len(previous_items),
    )
    extra = np.column_stack(
        [similarity, np.log1p(counts), similarity ** 2, np.maximum(similarity - 0.1, 0.0)]
    )
    return np.column_stack([features, extra])


def transition_counts(sources, destinations):
    counts = {}
    for left, right in zip(sources.tolist(), destinations.tolist()):
        key = (int(left), int(right))
        counts[key] = counts.get(key, 0) + 1
    return counts


def task6():
    print("Task 6: time-to-next interaction", flush=True)
    started = time.time()
    users, items, times, ratings, user_map, item_map = load_train(os.path.join(DATA_DIR, "task_6_train.csv"))
    order = np.lexsort((times, users))
    users, items, times, ratings = users[order], items[order], times[order], ratings[order]
    n_users, n_items = len(user_map), len(item_map)
    starts, ends = group_bounds(users)
    sources, destinations, gaps = consecutive_edges(users, items, times)
    destination_event = np.flatnonzero(users[1:] == users[:-1]) + 1
    final_edge = destination_event == (ends[users[destination_event]] - 1)
    train_sources = sources[~final_edge]
    train_destinations = destinations[~final_edge]
    train_gaps = gaps[~final_edge]
    item_typical, _ = item_gap_prior(train_destinations, train_gaps, n_items, GAP_PRIOR)
    prefix = np.ones(len(users), np.bool_)
    prefix[ends - 1] = False
    popularity = np.bincount(items[prefix], minlength=n_items).astype(np.float64)
    _, cosine = build_similarity(users[prefix], items[prefix], n_users, n_items)
    transitions = transition_counts(train_sources, train_destinations)
    print("  fitting holdout gap model", flush=True)
    features, labels, previous_items, destination_items = collect_gap_examples(
        starts, ends, items, times, ratings, item_typical, popularity, include_last=False
    )
    features = attach_book_features(features, cosine, transitions, previous_items, destination_items)
    weight, bias, center, scale = fit_gap_model(features, labels)

    val_features = []
    val_labels = []
    val_median = []
    val_previous = []
    val_destination = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        if end - start < 4:
            continue
        stamps = times[start:end]
        deltas = np.diff(stamps.astype(np.float64))
        index = len(deltas) - 1
        val_features.append(
            gap_feature_row(
                deltas[:index],
                np.log1p(deltas[:index]),
                stamps[index],
                ratings[start + index],
                item_typical[int(items[start + index + 1])],
                popularity[int(items[start + index + 1])],
            )
        )
        val_labels.append(float(deltas[index]))
        val_median.append(float(np.median(deltas[:index])))
        val_previous.append(int(items[start + index]))
        val_destination.append(int(items[start + index + 1]))
    val_features = np.asarray(val_features, np.float64)
    val_labels = np.asarray(val_labels, np.float64)
    val_features = attach_book_features(
        val_features,
        cosine,
        transitions,
        np.asarray(val_previous, np.int64),
        np.asarray(val_destination, np.int64),
    )
    prediction = predict_gap(val_features, weight, bias, center, scale)
    baseline_mae = float(np.mean(np.abs(np.asarray(val_median) - val_labels)))
    holdout_mae = float(np.mean(np.abs(prediction - val_labels)))
    print(f"  user-median MAE {baseline_mae:.0f}  model MAE {holdout_mae:.0f}", flush=True)

    print("  refitting on every gap", flush=True)
    full_typical, _ = item_gap_prior(destinations, gaps, n_items, GAP_PRIOR)
    full_popularity = np.bincount(items, minlength=n_items).astype(np.float64)
    _, full_cosine = build_similarity(users, items, n_users, n_items)
    full_transitions = transition_counts(sources, destinations)
    full_features, full_labels, full_previous, full_destination = collect_gap_examples(
        starts, ends, items, times, ratings, full_typical, full_popularity, include_last=True
    )
    # The destination of a training gap is inside the transition counts; drop that one count.
    seen = np.fromiter(
        (full_transitions.get((int(left), int(right)), 0) for left, right in zip(full_previous, full_destination)),
        np.float64,
        count=len(full_previous),
    )
    full_features = attach_book_features(
        full_features, full_cosine, full_transitions, full_previous, full_destination
    )
    count_column = full_features.shape[1] - 3
    full_features[:, count_column] = np.log1p(np.maximum(seen - 1.0, 0.0))
    weight, bias, center, scale = fit_gap_model(full_features, full_labels)

    header, rows = read_test_rows(os.path.join(DATA_DIR, "task_6_test.csv"))
    test_features = []
    test_keys = []
    previous_ids = []
    destination_ids = []
    for user_name, item_name in rows:
        user = user_map[user_name]
        start, end = int(starts[user]), int(ends[user])
        deltas = np.diff(times[start:end].astype(np.float64))
        destination = item_map.get(item_name, -1)
        typical = full_typical[destination] if destination >= 0 else float(np.median(gaps))
        item_popularity = full_popularity[destination] if destination >= 0 else 0.0
        test_features.append(
            gap_feature_row(
                deltas,
                np.log1p(deltas),
                times[end - 1],
                ratings[end - 1],
                typical,
                item_popularity,
            )
        )
        previous_ids.append(int(items[end - 1]))
        destination_ids.append(destination if destination >= 0 else 0)
        test_keys.append((user_name, item_name, destination >= 0))
    test_features = np.asarray(test_features, np.float64)
    previous_ids = np.asarray(previous_ids, np.int64)
    destination_ids = np.asarray(destination_ids, np.int64)
    known_destination = np.array([key[2] for key in test_keys])
    similarity = np.zeros(len(test_keys), np.float64)
    counts = np.zeros(len(test_keys), np.float64)
    if np.any(known_destination):
        similarity[known_destination] = pairwise_cosine(
            full_cosine, previous_ids[known_destination], destination_ids[known_destination]
        )
        counts[known_destination] = np.fromiter(
            (
                full_transitions.get((int(left), int(right)), 0)
                for left, right, keep in zip(previous_ids, destination_ids, known_destination)
                if keep
            ),
            np.float64,
            count=int(known_destination.sum()),
        )
    extra = np.column_stack([similarity, np.log1p(counts), similarity ** 2, np.maximum(similarity - 0.1, 0.0)])
    test_features = np.column_stack([test_features, extra])
    predicted = predict_gap(test_features, weight, bias, center, scale)
    lines = [
        f"{user_name},{item_name},{value:.1f}"
        for (user_name, item_name, _), value in zip(test_keys, predicted)
    ]
    write_predictions(os.path.join(DATA_DIR, "task_6_predictions.csv"), header, lines)
    print(f"  wrote {len(lines)} predictions in {time.time() - started:.1f}s", flush=True)
    return holdout_mae


def main():
    print(f"device {DEVICE}", flush=True)
    task1_mse = task1()
    task2_acc = task2()
    task3_mse = task3()
    task4_acc = task4()
    task5_acc = task5()
    task6_mae = task6()
    print(
        f"done  task1_mse={task1_mse:.4f} task2_acc={task2_acc:.4f} "
        f"task3_mse={task3_mse:.4f} task4_acc={task4_acc:.4f} "
        f"task5_acc={task5_acc:.4f} task6_mae={task6_mae:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
