"""Alembic environment. Reads DATABASE_URL from env (or .env)."""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise RuntimeError("DATABASE_URL not set. Copy .env.example to .env and fill in.")
config.set_main_option("sqlalchemy.url", db_url)

# Autogenerate is disabled for now — migrations are hand-written. When we move to
# SQLAlchemy models in M1b, set target_metadata here and enable --autogenerate.
target_metadata = None


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
