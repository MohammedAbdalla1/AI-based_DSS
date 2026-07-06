from reports import service as svc
from reports.models import NarrateSectionRequest, PlanReportRequest


def test_run_plan_report_falls_back_to_groq(monkeypatch):
    def _raise_gemini(*args, **kwargs):
        raise svc.SQLGenerationError("Gemini generation failed: RuntimeError", subcode="PROVIDER_FAILURE")

    monkeypatch.setattr(svc, "_generate_text_with_retry", _raise_gemini)
    monkeypatch.setattr(
        svc,
        "_generate_text_with_groq",
        lambda *args, **kwargs: (
            '{"title":"Sales Summary","summary":"Monthly overview","sections":[{"heading":"Revenue","sql":"SELECT 1","chart_type":"bar"}]}'
        ),
    )

    result = svc.run_plan_report(
        PlanReportRequest(
            db_type="postgres",
            role_schema={"orders": {"id": "INT"}},
            user_prompt="Summarize monthly revenue",
            max_sections=3,
        )
    )

    assert result.status == "success"
    assert result.title == "Sales Summary"
    assert result.sections[0].heading == "Revenue"


def test_run_narrate_section_falls_back_to_groq(monkeypatch):
    def _raise_gemini(*args, **kwargs):
        raise svc.SQLGenerationError("Gemini generation failed: RuntimeError", subcode="MODEL_UNAVAILABLE")

    monkeypatch.setattr(svc, "_generate_text_with_retry", _raise_gemini)
    monkeypatch.setattr(svc, "_generate_text_with_groq", lambda *args, **kwargs: "Groq wrote this narrative.")

    result = svc.run_narrate_section(
        NarrateSectionRequest(
            heading="Revenue",
            user_prompt="Summarize monthly revenue",
            sql="SELECT 1",
            sample_rows_json='[{"revenue": 100}]',
        )
    )

    assert result.status == "success"
    assert result.narrative == "Groq wrote this narrative."
