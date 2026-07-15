import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_fit_studio.requirements import parse_requirements


def test_parses_bulleted_lines():
    posting = """
    - Experience with Python and PyTorch
    - Familiarity with Docker and CI/CD pipelines
    - Strong background in computer vision
    """
    reqs = parse_requirements(posting)
    assert len(reqs) == 3
    assert "Python" in reqs[0]


def test_strips_bullet_markers_and_numbering():
    posting = "1. Experience with SQL databases\n* Familiarity with REST APIs\n• Team collaboration skills"
    reqs = parse_requirements(posting)
    assert reqs[0] == "Experience with SQL databases"
    assert reqs[1] == "Familiarity with REST APIs"
    assert reqs[2] == "Team collaboration skills"


def test_filters_out_short_fragments():
    posting = "Requirements:\n- Python experience required for this role\n-\n요구사항"
    reqs = parse_requirements(posting, min_words=3)
    assert "Requirements:" not in reqs
    assert "요구사항" not in reqs
    assert any("Python" in r for r in reqs)


def test_filters_boilerplate_section_headers():
    posting = "자격요건:\n- 3년 이상의 실무 경험\n우대사항:\n- 관련 자격증 보유"
    reqs = parse_requirements(posting)
    assert "자격요건:" not in reqs
    assert "우대사항:" not in reqs


def test_empty_posting_returns_empty_list():
    assert parse_requirements("") == []
    assert parse_requirements("   \n  \n  ") == []
