from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.preprocessing import MinMaxScaler,StandardScaler

preprocessor = ColumnTransformer([
    (
        "scaling",
        StandardScaler(),
        make_column_selector(dtype_include="number")
    )
])
