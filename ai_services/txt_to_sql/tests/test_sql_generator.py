from txt_to_sql.prompt_builder import build_sql_prompt
from txt_to_sql.schema_validator import build_role_schema
from txt_to_sql.sql_generator import generate_sql


def test_sql_generation():
    role = "sales"
    question = "get Monthly revenue trend per product category"

    schema = build_role_schema(role)
    prompt = build_sql_prompt(question, role, schema)

    sql = generate_sql(prompt)

    print("\n----- GENERATED SQL -----\n")
    print(sql)
    print("\n-------------------------\n")


if __name__ == "__main__":
    test_sql_generation()
