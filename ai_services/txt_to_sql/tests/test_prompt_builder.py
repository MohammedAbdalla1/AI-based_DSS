from txt_to_sql.prompt_builder import build_sql_prompt
from txt_to_sql.schema_validator import build_role_schema

def test_prompt_builder():
    role = "sales"
    question = "Total revenue per product category"

    schema = build_role_schema(role)
    prompt = build_sql_prompt(question=question, role=role, schema=schema)

    print("\n ----------generated prompt --------------")
    print(prompt)
    print("--------------------------------------------")

    assert question in prompt
    assert role in prompt


if __name__ == "__main__":
    test_prompt_builder()