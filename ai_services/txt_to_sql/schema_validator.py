"""
schema_validator.py

This module is responsible for:
- Validating role-based access definitions
- Deriving a role-specific schema from the FULL_SCHEMA
- Enforcing that role schemas are true subsets of the full schema

This logic runs in the backend.
The LLM never sees the FULL_SCHEMA unless the role is explicitly allowed.
"""

from txt_to_sql.schemas import FULL_SCHEMA, ROLE_TABLE_ACCESS


class SchemaValidationError(Exception):
    """Raised when a role schema definition is invalid."""
    pass


def build_role_schema(role: str) -> dict:
    """
    Build and return a role-specific schema.

    Args:
        role (str): user role (e.g. 'sales', 'marketing', 'inventory', 'admin')

    Returns:
        dict: schema limited to what the role is allowed to access

    Raises:
        SchemaValidationError: if the role or its schema definition is invalid
    """

    if role not in ROLE_TABLE_ACCESS:
        raise SchemaValidationError(f"Unknown role: '{role}'")

    role_access = ROLE_TABLE_ACCESS[role]

    # ---- Admin case: full schema access ----
    if role_access == "ALL":
        return FULL_SCHEMA

    role_schema = {}

    for table_name, allowed_columns in role_access.items():

        # Validate table existence
        if table_name not in FULL_SCHEMA:
            raise SchemaValidationError(
                f"Table '{table_name}' is not defined in FULL SCHEMA"
            )

        full_columns = FULL_SCHEMA[table_name]

        # Validate column existence
        for column in allowed_columns:
            if column not in full_columns:
                raise SchemaValidationError(
                    f"Column '{column}' does not exist in table '{table_name}'"
                )

        # Add validated table slice
        role_schema[table_name] = allowed_columns

    return role_schema
