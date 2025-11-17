import os
import pandas as pd

# Archivo encargado de definir funciones ETL reutilizables para los DAGs de Airflow


def create_folders(**kwargs):
    """
    Crea una carpeta con la fecha de ejecución (proveniente del DAG)
    y tres subcarpetas: 'raw', 'splits' y 'models'.

    Parameters:
        **kwargs: Recibe los parámetros del contexto de Airflow.
                  Usa 'ds' (date string, formato YYYY-MM-DD).
    """
    # Obtener la fecha de ejecución desde Airflow (YYYY-MM-DD)
    execution_date = kwargs.get("ds")

    # Crear carpeta principal
    base_folder = execution_date
    os.makedirs(base_folder, exist_ok=True)

    # Crear subcarpetas
    subfolders = ["raw", "preprocessed", "feature_extracted", "splits", "models", "predictions"]
    for sub in subfolders:
        os.makedirs(os.path.join(base_folder, sub), exist_ok=True)

    print(f"Carpetas creadas en: {os.path.abspath(base_folder)}")
    return {
        "raw_data_path": os.path.abspath(os.path.join(base_folder, "raw")),
        "splits_path": os.path.abspath(os.path.join(base_folder, "splits")),
        "models_path": os.path.abspath(os.path.join(base_folder, "models")),
    }


def transform_data(fast_debug=True, only_last_week=False, last_week_path=None, **kwargs):
    """
    Realiza las transformaciones necesarias en los datos para el modelo de recomendación.

    """
    # Obtener fecha desde Airflow (si se pasa como contexto)
    execution_date = kwargs.get("ds")

    # Definir rutas de entrada y salida
    base_folder = execution_date
    client_path = os.path.join(base_folder, "raw", "clientes.parquet")
    product_path = os.path.join(base_folder, "raw", "productos.parquet")

    if only_last_week:
        transaction_path = os.path.join(base_folder, "raw", "new_transactions.parquet")
    else:
        transaction_path = os.path.join(base_folder, "raw", "transacciones.parquet")

    # Lectura de datos
    clients = pd.read_parquet(client_path)
    products = pd.read_parquet(product_path)
    transactions = pd.read_parquet(transaction_path)

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

    print(f"last_week_path: {last_week_path}")
    # Merge con datos de la semana pasada si es necesario
    if only_last_week and last_week_path is not None:
        last_week_df = pd.read_parquet(last_week_path)
        main_df = pd.concat([last_week_df, main_df], ignore_index=True)
        main_df = main_df.drop_duplicates(keep="last", subset=["customer_id", "product_id", "purchase_date"])

    # Guardado de datos transformados semanalmente
    preprocessed_folder = os.path.join(base_folder, "preprocessed")
    os.makedirs(preprocessed_folder, exist_ok=True)

    output_path = os.path.join(preprocessed_folder, "weekly_data.parquet")
    main_df.to_parquet(output_path, index=False)

    # Print de confirmación
    print(f"Datos preprocesados guardados en: {os.path.abspath(output_path)}")
    print(f"Dimensiones del dataset preprocesado: {main_df.shape}")
    print(f"Columnas del dataset preprocesado: {main_df.columns.tolist()}")
    print(f"Head del dataset preprocesado:\n{main_df.head()}")
    print(f"Maxima fecha en el dataset preprocesado: {main_df['purchase_date'].max()}")


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
        .resample("W-MON", label="left", closed="left")
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


def feature_lagger(df):
    """
    Agrega características rezagadas (lagged features) al dataframe proporcionado.
    """

    df = df.sort_values(by=["customer_id", "product_id", "purchase_date"])

    # Definir los lags a crear
    lag_weeks = [1, 2, 3, 4]
    for lag in lag_weeks:
        df[f"num_orders_lag_{lag}"] = df.groupby(["customer_id", "product_id"])["num_orders"].shift(lag).copy()
        df[f"items_lag_{lag}"] = df.groupby(["customer_id", "product_id"])["items"].shift(lag).copy()

    # Eliminar columnas sin lags, ya que son data leaks
    df = df.drop(columns=["num_orders", "items"])

    return df


def training_lagged_features(**kwargs):
    """
    Agrega características rezagadas (lagged features) a los datos preprocesados.
    """
    # Obtener fecha desde Airflow (si se pasa como contexto)
    execution_date = kwargs.get("ds")

    # Definir rutas de entrada y salida
    base_folder = execution_date
    preprocessed_folder = os.path.join(base_folder, "preprocessed")
    feature_extracted_folder = os.path.join(base_folder, "feature_extracted")
    os.makedirs(feature_extracted_folder, exist_ok=True)

    # Leer el dataset preprocesado
    df = pd.read_parquet(os.path.join(preprocessed_folder, "weekly_data.parquet"))

    # Crear lagged features
    df = feature_lagger(df)

    # Guardado de datos con lags
    output_path = os.path.join(feature_extracted_folder, "weekly_data_with_lags.parquet")
    df.to_parquet(output_path, index=False)


def split_data(**kwargs):
    """
    Lee 'datos semanalmente preprocesados' desde la carpeta 'feature_extracted',
    realiza hold-out split (80/20) y guarda los datasets resultantes en 'splits'.
    Parameters:
        **kwargs: permite obtener la fecha de ejecución desde Airflow (ds)
    """
    # Definir rutas de entrada y salida
    execution_date = kwargs.get("ds")
    base_folder = execution_date
    feature_extracted_folder = os.path.join(base_folder, "feature_extracted")
    splits_folder = os.path.join(base_folder, "splits")
    os.makedirs(splits_folder, exist_ok=True)

    # Leer el dataset preprocesado
    df = pd.read_parquet(os.path.join(feature_extracted_folder, "weekly_data_with_lags.parquet"))

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


def next_week_data(**kwargs):
    """
    Genera y guarda los datos para la semana siguiente.
    """
    # Obtener fecha desde Airflow (si se pasa como contexto)
    execution_date = kwargs.get("ds")

    # Definir rutas de entrada y salida
    base_folder = execution_date
    preprocessed_folder = os.path.join(base_folder, "preprocessed")
    feature_extracted_folder = os.path.join(base_folder, "feature_extracted")
    os.makedirs(feature_extracted_folder, exist_ok=True)

    # Leer el dataset preprocesado
    main_df = pd.read_parquet(os.path.join(preprocessed_folder, "weekly_data.parquet"))
    main_df = main_df.sort_values(by=["customer_id", "product_id", "purchase_date"])

    # Crear filas para los pares customer-product para la semana siguiente
    last_date = main_df["purchase_date"].max()
    next_week_date = last_date + pd.Timedelta(weeks=1)

    def create_next_week_rows(group):
        last_row = group.iloc[-1]
        new_row = last_row.copy()
        new_row["purchase_date"] = next_week_date
        new_row["items"] = 0
        new_row["num_orders"] = 0
        return pd.DataFrame([new_row])

    next_week_df = main_df.groupby(["customer_id", "product_id"], group_keys=False).apply(create_next_week_rows)
    main_df = pd.concat([main_df, next_week_df], ignore_index=True)

    # Crear lagged features
    main_df = feature_lagger(main_df)

    # Filtrar solo la semana siguiente
    next_week_data = main_df[main_df["purchase_date"] == next_week_date]

    # Guardado de datos con lags para la semana siguiente
    output_path = os.path.join(feature_extracted_folder, "next_week_data.parquet")
    next_week_data.to_parquet(output_path, index=False)
