from pathlib import Path
from sqlalchemy import create_engine


def save_tables_to_sqlite(db_path, tables):
    """
    Saves project tables to a SQLite database.
    """

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{db_path}")

    for table_name, df in tables.items():
        clean_df = df.copy()

        clean_df.to_sql(
            table_name,
            engine,
            if_exists="replace",
            index=False,
        )