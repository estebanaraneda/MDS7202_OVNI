import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import gradio as gr

# Archivo encargado de definir funciones ETL reutilizables para los DAGs de Airflow


def extract_data():
    """
    Por ahora lee los datos de una carperta 'data'.

    """
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    clients = pd.read_parquet(os.path.join(data_folder, "clientes.parquet"))
    products = pd.read_parquet(os.path.join(data_folder, "productos.parquet"))
    transactions = pd.read_parquet(os.path.join(data_folder, "transacciones.parquet"))

    return clients, products, transactions


def extract_new_data():
    """
    Lee nuevos datos de transacciones desde la carpeta 'new_data'.
    """
    data_folder = "new_data"
    os.makedirs(data_folder, exist_ok=True)

    new_transactions = pd.read_parquet(os.path.join(data_folder, "new_transacciones.parquet"))

    return new_transactions


def transform_data(clients, products, transactions, fast_debug=True):
    """
    Realiza las transformaciones necesarias en los datos para el modelo de recomendación.

    """
    # Drop de duplicados
    clients = clients.drop_duplicates(keep="first")
    products = products.drop_duplicates(keep="first")
    transactions = transactions.drop_duplicates(keep="first")

    # Merge de dataframes
    main_df = transactions.merge(clients, on="customer_id", how="left").merge(products, on="product_id", how="left")

    # Eliminar columnas no relevantes
    main_df = main_df.drop(columns=["num_visit_per_week", "region_id", "zone_id", "category", "sub_category"])

    # Seteo de variables categóricas
    main_df["brand"] = main_df["brand"].astype("category")
    main_df["segment"] = main_df["segment"].astype("category")
    main_df["package"] = main_df["package"].astype("category")
    main_df["customer_type"] = main_df["customer_type"].astype("category")
    # Esta última se vio en el EDA por lo que la dejamos como categórica
    main_df["num_deliver_per_week"] = main_df["num_deliver_per_week"].astype("category")

    # Diccionario de tipo de datos para cada columna
    # para consistencia futura
    dtype_dict = {}
    for col in main_df.columns:
        col_type = main_df[col].dtype
        dtype_dict[col] = col_type

    print(dtype_dict)

    if fast_debug:
        random_clients_ids = main_df["customer_id"].drop_duplicates().sample(n=5, random_state=42)
        main_df = main_df[main_df["customer_id"].isin(random_clients_ids)]

    # Transformación y reindexado por semana
    main_df = dataframe_reindexer(main_df)

    # Recuperación de tipos originales
    for col, col_type in dtype_dict.items():
        if col in main_df.columns:
            if col != "purchase_date":  # Evitar conflictos con datetime
                main_df[col] = main_df[col].astype(col_type)
            else:
                main_df["purchase_date"] = main_df["purchase_date"].dt.tz_localize(None)

    # Sorteo de datos por cliente, producto y fecha
    main_df.sort_values(by=["customer_id", "product_id", "purchase_date"], inplace=True)

    # Tranformar customer_id y product_id a categóricas
    main_df["customer_id"] = main_df["customer_id"].astype("category")
    main_df["product_id"] = main_df["product_id"].astype("category")
    return main_df


def weekly_data_saver(df):
    """
    Guarda el dataframe transformado en la carpeta 'preprocessed' con el nombre 'weekly_data.csv'.
    """
    preprocessed_folder = "preprocessed"
    os.makedirs(preprocessed_folder, exist_ok=True)

    output_path = os.path.join(preprocessed_folder, "weekly_data.parquet")
    df.to_parquet(output_path, index=False)

    print(f"Data semanal guardada en: {output_path}")


def reindex_customer_product(group, all_dates):
    """
    Reindexa un grupo específico de customer_id y product_id con todas las fechas posibles de comienzo de semana.
    """
    # Reindexa el grupo con todas las fechas necesarias
    group = group.set_index("purchase_date").reindex(all_dates)

    # En las fechas donde no hubo mathches entre purchase_date y all_dates
    # se crea una fila con NaNs, las cuales se deben rellenar adecuadamente
    # Para items y num_orders se rellenan con 0, ya que no hubo compras esa semana
    group["items"] = group["items"].fillna(0)
    group["num_orders"] = group["num_orders"].fillna(0)
    # Para las demás columnas se usa forward fill y backward fill,
    # ya que se asume que la información del cliente/producto no cambia
    group = group.ffill().bfill()
    # Reset index y renombrar columna de fecha
    group = group.reset_index().rename(columns={"index": "purchase_date"})
    return group


def dataframe_reindexer(df):
    """
    Reindexa el dataframe para asegurar que todas las combinaciones de customer_id, product_id y semanas estén.
    Agrupa por customer_id y product_id, y reindexa cada grupo con todas las fechas de comienzo de semana disponibles.
    """

    # Indexado por fecha
    df.set_index("purchase_date", inplace=True)

    # Resampleo semanal
    weekly_df = (
        df.groupby(["customer_id", "product_id"])
        .resample("W-MON")
        .agg(
            {
                "order_id": "nunique",  # Número de órdenes
                "items": "sum",  # Total pagado
                "customer_type": "first",  # Mantener tipo de cliente
                "Y": "first",  # Mantener coordenada Y
                "X": "first",  # Mantener coordenada X
                "num_deliver_per_week": "first",  # Mantener número de entregas por semana
                "brand": "first",  # Mantener marca
                "segment": "first",  # Mantener segmento
                "package": "first",  # Mantener paquete
                "size": "first",  # Mantener tamaño
            }
        )
        .reset_index()
    )

    weekly_df.rename(columns={"order_id": "num_orders"}, inplace=True)
    weekly_df["purchase_date"] = pd.to_datetime(weekly_df["purchase_date"], utc=True)

    # Todas las fechas que deben existir
    all_dates = pd.date_range(
        start=weekly_df["purchase_date"].min(), end=weekly_df["purchase_date"].max(), freq="W-MON"
    ).tz_convert("UTC")

    # Fechas a timestamp
    all_dates = pd.to_datetime(all_dates)

    # Se crean todas las fechas para cada customer_id y product_id, ya que actualmente
    # no hay filas para fechas sin órdenes
    weekly_df = weekly_df.groupby(["customer_id", "product_id"], group_keys=False).apply(
        lambda g: reindex_customer_product(g, all_dates), include_groups=True
    )

    # Transformación de num_orders a 0 y 1
    weekly_df["order"] = weekly_df["num_orders"].apply(lambda x: 1 if x > 0 else 0)

    # Asegurar que purchase_date es datetime
    weekly_df["purchase_date"] = pd.to_datetime(weekly_df["purchase_date"], utc=True)

    return weekly_df


def lagged_features_adder(df, lag_weeks=[1, 2, 3, 4]):
    """
    Agrega características rezagadas (lagged features) al dataframe.
    Crea columnas con el número de órdenes en semanas anteriores.
    """
    df = df.sort_values(by=["customer_id", "product_id", "purchase_date"])

    for lag in lag_weeks:
        df[f"num_orders_lag_{lag}"] = df.groupby(["customer_id", "product_id"])["num_orders"].shift(lag)
        df[f"items_lag_{lag}"] = df.groupby(["customer_id", "product_id"])["items"].shift(lag)

    return df


def split_data(**kwargs):
    """
    Lee 'datos semanalmente preprocesados' desde la carpeta 'preprocessed',
    realiza hold-out split (80/20) y guarda los datasets resultantes en 'splits'.
    Parameters:
        **kwargs: permite obtener la fecha de ejecución desde Airflow (ds)
    """
    # Definir rutas de entrada y salida
    preprocessed_folder = "preprocessed"
    splits_folder = "splits"
    os.makedirs(splits_folder, exist_ok=True)
    # Leer el dataset preprocesado
    data_path = os.path.join(preprocessed_folder, "weekly_data.parquet")
    df = pd.read_parquet(data_path)
    # Starting date
    initial_date = df["purchase_date"].min()
    final_date = df["purchase_date"].max()
    # Definir fecha de corte para hold-out split (60/20/20)
    cutoff_date = initial_date + (final_date - initial_date) * 0.6
    test_cutoff_date = cutoff_date + (final_date - initial_date) * 0.2
    # Dividir los datos
    train_df = df[df["purchase_date"] <= cutoff_date]
    val_df = df[(df["purchase_date"] > cutoff_date) & (df["purchase_date"] <= test_cutoff_date)]
    test_df = df[df["purchase_date"] > test_cutoff_date]
    # Guardar los splits
    train_df.to_parquet(os.path.join(splits_folder, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(splits_folder, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(splits_folder, "test.parquet"), index=False)
    print(f"Conjuntos guardados en: {os.path.abspath(splits_folder)}")


# Zona de testing
if __name__ == "__main__":
    # Prueba de las funciones ETL
    clients, products, transactions = extract_data(ds="2024-06-01")
    main_df = transform_data(clients, products, transactions, fast_debug=True)
    print(main_df.head())
