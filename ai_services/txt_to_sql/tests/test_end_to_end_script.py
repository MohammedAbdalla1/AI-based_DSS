from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

# Ensure `from txt_to_sql...` imports work when running:
# python tests/test_end_to_end_script.py
AI_SERVICES_DIR = Path(__file__).resolve().parents[2]
if str(AI_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICES_DIR))

from txt_to_sql.models import TextToSQLRequest
from txt_to_sql.prompt_builder import build_sql_prompt
from txt_to_sql.sql_generator import SQLGenerationError, generate_sql
from txt_to_sql.sql_validator import SQLValidationError, validate_sql


DEFAULT_PAYLOAD = {
    "question": "For each month in the last 12 full months, return total orders, unique customers, average order value, and cumulative revenue over time.",
    "db_type": "postgres",
    "role_schema": {
        "products": {
            "id": "INT",
            "name": "TEXT",
            "price": "DECIMAL",
            "category_id": "INT",
        },
        "orders": {
            "id": "INT",
            "customer_id": "INT",
            "order_date": "DATE",
            "total_amount": "DECIMAL",
        },
        "order_items": {
            "id": "INT",
            "order_id": "INT",
            "product_id": "INT",
            "quantity": "INT",
            "unit_price": "DECIMAL",
        },
        "customers": {
            "id": "INT",
            "full_name": "TEXT",
            "email": "TEXT",
            "city": "TEXT",
        },
    },
}


def main() -> int:
    service_dir = Path(__file__).resolve().parents[1]
    env_path = service_dir / ".env"
    load_dotenv(dotenv_path=env_path)

    print("=== TXT-to-SQL End-to-End Script ===")
    print(f".env path: {env_path}")

    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is missing.")
        return 1

    generated_sql = None

    try:
        request = TextToSQLRequest(**DEFAULT_PAYLOAD)
        prompt = build_sql_prompt(request.question, request.role_schema, request.db_type)
        generated_sql = generate_sql(prompt)

        print("\nGenerated SQL (raw from Gemini):")
        print(generated_sql)

        validated_sql = validate_sql(generated_sql, request.role_schema, request.db_type)
    except ValidationError as exc:
        print("REQUEST_VALIDATION_ERROR")
        print(exc)
        return 2
    except SQLGenerationError as exc:
        print("SQL_GENERATION_ERROR")
        print(exc)
        return 3
    except SQLValidationError as exc:
        print("SQL_VALIDATION_ERROR")
        if generated_sql is not None:
            print("\nGenerated SQL (raw from Gemini):")
            print(generated_sql)
        print(exc)
        return 4
    except Exception as exc:
        print("UNEXPECTED_ERROR")
        print(f"{type(exc).__name__}: {exc}")
        return 5

    print("\nRequest:")
    print(request.model_dump())
    print("\nPrompt (first 500 chars):")
    print(prompt[:500])
    print("\nGenerated SQL:")
    print(generated_sql)
    print("\nValidated SQL:")
    print(validated_sql)
    print("\nSUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
