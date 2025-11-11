# Archivo de constantes para el DAG de Airflow

SEED = 42

CATEGORICAL_VARIABLES = [
    "customer_id",
    "product_id",
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
    "payment_lagged_1",
    "payment_lagged_2",
    "payment_lagged_3",
    "payment_lagged_4",
    "num_orders_lagged_1",
    "num_orders_lagged_2",
    "num_orders_lagged_3",
    "num_orders_lagged_4",
]

TEMPORAL_VARIABLES = [
    "day",
    "month",
    "week",
]
