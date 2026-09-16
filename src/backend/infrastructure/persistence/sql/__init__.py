"""PostgreSQL persistence implementation.

SQL modules are imported lazily by the dependency container so the default
memory backend can still run in environments that have not installed database
dependencies yet.
"""
