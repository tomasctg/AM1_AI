from Datasets import get_wine_dataset, label_encode
from Benchmarking import Benchmark
from QDA import QDA, TensorizedQDA
from LDA import LDA

X_full, y_full = get_wine_dataset()
y_full_encoded = label_encode(y_full)

b = Benchmark(
    X_full, y_full_encoded,
    n_runs=100,
    warmup=20,
    mem_runs=20,
    test_sz=0.3,
    same_splits=False
)

to_bench = [QDA, TensorizedQDA, LDA]

for model in to_bench:
    b.bench(model)

summ = b.summary(baseline='QDA')
print(summ[[
    'train_median_ms', 'test_median_ms', 'mean_accuracy',
    'train_speedup', 'test_speedup',
    'train_mem_reduction', 'test_mem_reduction'
]])
