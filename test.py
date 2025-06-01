from Datasets import get_wine_dataset, label_encode, split_transpose
from numpy.random import RandomState
from QDA import QDA, TensorizedQDA
from LDA import LDA

X_full, y_full = get_wine_dataset()

X_full.shape, y_full.shape

y_full_encoded = label_encode(y_full)

X_train, X_test, y_train, y_test = split_transpose(
                X_full, y_full_encoded,0.3,RandomState(42)
            )

# qda_model = QDA()
# qda_model.fit(X_train,y_train)
# qda_model.predict(X_test)

# lda_model = LDA()
# lda_model.fit(X_train,y_train)
# lda_model.predict(X_test)

lda_model = TensorizedQDA()
lda_model.fit(X_train,y_train)
lda_model.predict(X_test)


