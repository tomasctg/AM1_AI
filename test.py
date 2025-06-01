from numpy.random import RandomState
from Benchmarking import Benchmark
from Datasets import get_wine_dataset, label_encode, split_transpose
from QDA import QDA, TensorizedQDA, FasterQDA
from LDA import LDA

X_full, y_full = get_wine_dataset()
y_full_encoded = label_encode(y_full)

# X_train, X_test, y_train, y_test = split_transpose(
#                 X_full, y_full_encoded,0.3,RandomState(42)
#             )

# qda_model = FasterQDA()
# qda_model.fit(X_train,y_train)
# qda_model.predict(X_test)

# qda_model = QDA()
# qda_model.fit(X_train,y_train)
# qda_model.predict(X_test)

# lda_model = LDA()
# lda_model.fit(X_train,y_train)
# lda_model.predict(X_test)

# lda_model = TensorizedQDA()
# lda_model.fit(X_train,y_train)
# lda_model.predict(X_test)


b = Benchmark(
    X_full, y_full_encoded,
    n_runs=100,
    warmup=20,
    mem_runs=20,
    test_sz=0.3,
    same_splits=False
)

to_bench = [QDA, TensorizedQDA, LDA, FasterQDA]

for model in to_bench:
    b.bench(model)

summ = b.summary(baseline='QDA')
print(summ[[
    'train_median_ms', 'test_median_ms', 'mean_accuracy',
    'train_speedup', 'test_speedup',
    'train_mem_reduction', 'test_mem_reduction'
]])
