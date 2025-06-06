from numpy.random import RandomState
from Benchmarking import Benchmark
from Datasets import get_wine_dataset, label_encode, split_transpose
from QDA import QDA, TensorizedQDA, FasterQDA, EfficientQDA, QDA_Chol1, QDA_Chol2, QDA_Chol3
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

to_bench = [QDA, TensorizedQDA, FasterQDA, EfficientQDA, QDA_Chol1, QDA_Chol2, QDA_Chol3]

for model in to_bench:
    b.bench(model)

summ = b.summary(baseline='QDA')
print(summ[[
    'test_median_ms', 'mean_accuracy',
    'test_speedup',
    'test_mem_reduction'
]])
