# Archivo de constantes para el DAG de Airflow

SEED = 42

CATEGORICAL_VARIABLES = [
    "customer_type",
    "num_deliver_per_week",
    "brand",
    "segment",
    "package",
]

NUMERICAL_VARIABLES = [
    "Y",
    "X",
    "size",
    "items_lag_1",
    "items_lag_2",
    "items_lag_3",
    "items_lag_4",
    "num_orders_lag_1",
    "num_orders_lag_2",
    "num_orders_lag_3",
    "num_orders_lag_4",
]

TEMPORAL_VARIABLES = [
    "day",
    "month",
    "week",
]
