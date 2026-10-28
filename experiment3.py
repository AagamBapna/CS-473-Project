"""CS473 Experiment 3: KNN with question-balanced clustering.

Run: python3 experiment3_updated.py training_dataset_updated.csv
Quick test: python3 experiment3_updated.py training_dataset_updated.csv --repeats 1 --clusters 2 5
Requires approach1.py in the same directory.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import approach1


# STEP 1: Read the training dataset
def read_data(path):
    df = pd.read_csv(path)

    df = df.rename(columns={
        'Id': 'id',
        'EssaySet': 'question',
        'Score1': 'score',
        'EssayText': 'text'
    })

    df = df[
        ['id', 'question', 'score', 'text']
    ].dropna(subset=['score']).copy()

    df['id'] = df['id'].astype(str)
    df['question'] = df['question'].astype(str)
    df['text'] = df['text'].fillna('').astype(str)
    df['score'] = pd.to_numeric(
        df['score'], errors='raise'
    )

    if df['id'].duplicated().any():
        raise ValueError('Submission IDs must be unique')

    return df.reset_index(drop=True)


def make_split(df, seed, test_size=0.1):
    """Split each question independently, guaranteeing train/test coverage."""
    train_parts, test_parts = [], []
    for _, group in df.groupby('question', sort=True):
        if len(group) < 2:
            raise ValueError('Each question needs at least two graded examples')
        train, test = train_test_split(group, test_size=test_size, random_state=seed)
        train_parts.append(train)
        test_parts.append(test)
    return (pd.concat(train_parts).reset_index(drop=True),
            pd.concat(test_parts).reset_index(drop=True))


def cluster_labels(pool, n_clusters):
    labels = np.empty(len(pool), dtype=object)
    questions = pool['question'].to_numpy()
    for q in pool['question'].unique():
        idx = np.flatnonzero(questions == q)
        texts = pool.iloc[idx]['text'].tolist()
        if len(idx) < 2 or not any(t.strip() for t in texts):
            labels[idx] = [f'{q}:0'] * len(idx)
            continue
        vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                     token_pattern=r'(?u)\b\w+\b')
        X = vectorizer.fit_transform(texts)

        if q == '1':
            print("Question:", q)
            print("Matrix shape:", X.shape)
            print("Minimum TF-IDF value:", X.data.min())
            print("Maximum TF-IDF value:", X.data.max())
            print("Contains NaN:", np.isnan(X.data).any())
            print("Contains infinity:", np.isinf(X.data).any())
            print("All-zero responses:", np.sum(X.getnnz(axis=1) == 0))

        if not np.isfinite(X.data).all():
            raise ValueError(f'Non-finite TF-IDF values for question {q}')
        k = min(n_clusters, len(idx))
        if k <= 1:
            labels[idx] = [f'{q}:0'] * len(idx)
            continue
        
        import warnings

        X_cluster = X.toarray().astype(np.float64)

        model = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=1,
            random_state=42,
            algorithm='lloyd'
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            fitted = model.fit_predict(X.astype(np.float64))

        for warning in caught:
            print(
                f"K-Means warning: question={q}, "
                f"N={k}, samples={X.shape[0]}, "
                f"features={X.shape[1]}, "
                f"message={warning.message}"
            )
        
        if not np.isfinite(model.cluster_centers_).all() or not np.isfinite(model.inertia_):
            raise ValueError(f'Non-finite K-Means result for question {q}, N={k}')
        labels[idx] = [f'{q}:{c}' for c in fitted]
    return labels


def allocate_counts(pool, size):
    """Allocate exactly size slots proportionally, with one per question."""
    counts = pool['question'].value_counts().sort_index()
    if size < len(counts):
        raise ValueError('Subset size is smaller than the number of questions')
    allocation = pd.Series(1, index=counts.index, dtype=int)
    remaining = size - len(counts)
    while remaining:
        eligible = counts.index[allocation < counts]
        if len(eligible) == 0:
            break
        # Give the next slot to the most underrepresented question relative to target.
        target = size * counts[eligible] / len(pool)
        q = (target - allocation[eligible]).idxmax()
        allocation[q] += 1
        remaining -= 1
    return allocation


def sample_subset(pool, labels, size, seed, clustered):
    rng = np.random.default_rng(seed)
    allocation = allocate_counts(pool, size)
    questions = pool['question'].to_numpy()
    chosen = []
    for q, q_size in allocation.items():
        idx = np.flatnonzero(questions == q)
        if not clustered:
            chosen.extend(rng.choice(idx, size=q_size, replace=False).tolist())
            continue
        groups = [rng.permutation(idx[labels[idx] == label]).tolist()
                  for label in sorted(set(labels[idx]))]
        local = []
        while len(local) < q_size:
            progress = False
            for g in rng.permutation(len(groups)):
                if groups[g] and len(local) < q_size:
                    local.append(groups[g].pop())
                    progress = True
            if not progress:
                break
        chosen.extend(local)
    result = np.sort(np.asarray(chosen, dtype=int))
    if len(result) != size or len(np.unique(result)) != size:
        raise RuntimeError('Sampling failed to select the requested number of unique rows')
    return result

def encode_grades(actual, pred, grade_labels):
    grades = np.asarray(
        sorted(set(grade_labels)),
        dtype=float
    )

    if len(grades) < 2:
        return None

    # Map each score to the nearest allowed grade
    def encode(values):
        values = np.asarray(values, dtype=float)
        return np.abs(
            values[:, None] - grades[None, :]
        ).argmin(axis=1)

    return encode(actual), encode(pred), np.arange(len(grades))

def score_subset(pool, test, indices, allowed):
    selected = pool.iloc[indices].to_dict('records')

    model = approach1.fit(selected)
    predictions = model.predict(test.to_dict('records'), allowed)

    pred_by_id = {
        str(p['id']): p['score']
        for p in predictions
    }

    pred = np.asarray([
        pred_by_id[str(i)]
        for i in test['id']
    ])

    actual = test['score'].to_numpy()
    questions = test['question'].to_numpy()

    scores = []

    for q in test['question'].unique():
        mask = questions == q

        if mask.sum() < 2:
            continue

        grade_labels = allowed[q]['allowed_grades']

        encoded = encode_grades(
            actual[mask],
            pred[mask],
            grade_labels
        )

        if encoded is None:
            # Single-grade questions are excluded consistently in all runs.
            continue

        actual_encoded, pred_encoded, labels_encoded = encoded

        # When both vectors are constant, sklearn's kappa may be undefined.
        # Apply an explicit convention rather than silently dropping the question.
        if np.unique(actual_encoded).size == 1 and np.unique(pred_encoded).size == 1:
            value = 1.0 if actual_encoded[0] == pred_encoded[0] else 0.0
        else:
            value = cohen_kappa_score(
                actual_encoded,
                pred_encoded,
                labels=labels_encoded,
                weights='quadratic'
            )

        if not np.isfinite(value):
            raise ValueError(f'Unexpected undefined QWK for question {q}')
        scores.append(value)

    return float(np.mean(scores)) if scores else np.nan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='Updated labeled training CSV')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Number of independently seeded validation splits')
    parser.add_argument('--clusters', type=int, nargs='+', default=[2, 3, 5, 10, 15])
    parser.add_argument('--out', default='experiment3_results')
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.clusters):
        parser.error('Repeats and cluster counts must be positive')
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    df = read_data(args.csv)
    # Prefer official rubric grades if available. Otherwise use known dataset grade set.
    allowed = {q: {'allowed_grades': sorted(g['score'].unique().tolist())}
               for q, g in df.groupby('question')}
    records = []
    for repeat in range(args.repeats):
        pool, test = make_split(df, seed=42 + repeat)
        n_questions = pool['question'].nunique()
        sizes = sorted(set(max(n_questions, int(round(len(pool) * p)))
                           for p in np.arange(0.1, 1.0, 0.1)))
        print(f'Split {repeat + 1}: {len(pool)} training, {len(test)} validation; '
              f'{n_questions} questions', flush=True)
        for n in args.clusters:
            labels = cluster_labels(pool, n)
            for size in sizes:
                chosen = sample_subset(pool, labels, size, seed=100 + repeat,
                                       clustered=True)
                qwk = score_subset(pool, test, chosen, allowed)
                records.append({'method': 'clustered', 'N': n, 'size': size,
                                'fraction': size / len(pool), 'repeat': repeat,
                                'qwk': qwk})
                print(f'N={n}, size={size}, repeat={repeat + 1}, QWK={qwk:.4f}',
                      flush=True)
        for size in sizes:
            chosen = sample_subset(pool, None, size, seed=100 + repeat,
                                   clustered=False)
            qwk = score_subset(pool, test, chosen, allowed)
            records.append({'method': 'random', 'N': np.nan, 'size': size,
                            'fraction': size / len(pool), 'repeat': repeat,
                            'qwk': qwk})
            print(f'Random size={size}, repeat={repeat + 1}, QWK={qwk:.4f}',
                  flush=True)
    result = pd.DataFrame(records)
    result.to_csv(out / 'all_results.csv', index=False)
    clustered = result[result['method'] == 'clustered']
    means = clustered.groupby(['N', 'fraction'])['qwk'].mean().unstack('fraction')
    means.to_csv(out / 'heatmap_values.csv')
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(means.to_numpy(), aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(means.columns)), [f'{100 * x:.0f}%' for x in means.columns])
    ax.set_yticks(range(len(means.index)), [int(x) for x in means.index])
    ax.set_xlabel('Fraction of training pool selected')
    ax.set_ylabel('Clusters per question (N)')
    ax.set_title('Clustered KNN: Mean QWK by Training Fraction and N')
    fig.colorbar(im, ax=ax, label='QWK')
    fig.tight_layout()
    fig.savefig(out / 'heatmap.png', dpi=180)
    plt.close(fig)
    by_n = clustered.groupby('N')['qwk'].mean().dropna()
    if by_n.empty:
        print('All clustered QWK scores were undefined; inspect grade distributions.')
        return
    best_n = int(by_n.idxmax())
    print(f'Selected N={best_n}')
    selected = clustered[clustered['N'] == best_n].groupby('fraction')['qwk'].mean()
    baseline = result[result['method'] == 'random'].groupby('fraction')['qwk'].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(baseline.index * 100, baseline.values, marker='o', label='Random selection')
    ax.plot(selected.index * 100, selected.values, marker='o',
            label=f'Cluster selection (N={best_n})')
    ax.set_xlabel('Percentage of training pool selected')
    ax.set_ylabel('Mean per-question QWK')
    ax.set_title('Experiment 3: KNN With vs. Without Clustering')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=180)
    plt.close(fig)
    print(f'Done! Results saved in {out.resolve()}')


if __name__ == '__main__':
    main()
