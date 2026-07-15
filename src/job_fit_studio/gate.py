"""
The capability gate: the honest branch point of the whole system.

Rather than always generating a cover letter regardless of how well you
actually match a posting, this checks the FitReport's coverage against a
minimum bar. Below the bar, drafting a confident-sounding cover letter
for a role you're not ready for would be actively dishonest (and would
read as generic/hollow to any real recruiter, since it wouldn't be
grounded in real matched evidence). Above the bar, there's enough real,
matched capability to write a genuinely evidence-based application.

This is deliberately a HARD gate with a visible reason, not a soft
suggestion -- the whole point is that the system refuses to oversell you.
"""
from dataclasses import dataclass

from .matcher import FitReport


@dataclass
class GateDecision:
    proceed_to_draft: bool
    coverage: float
    reason: str
    unmatched_requirements: list  # only populated when proceed_to_draft is False


def evaluate_gate(report: FitReport, min_coverage: float = 0.6) -> GateDecision:
    if report.coverage >= min_coverage:
        return GateDecision(
            proceed_to_draft=True,
            coverage=report.coverage,
            reason=(f"{report.coverage:.0%} of requirements matched real, evidenced capability "
                     f"(>= {min_coverage:.0%} threshold) -- proceeding to draft."),
            unmatched_requirements=[],
        )

    unmatched = [m.requirement for m in report.unmatched_requirements]
    return GateDecision(
        proceed_to_draft=False,
        coverage=report.coverage,
        reason=(f"Only {report.coverage:.0%} of requirements matched real, evidenced capability "
                 f"(below the {min_coverage:.0%} threshold) -- recommending skill/project gaps "
                 f"instead of drafting an underqualified application."),
        unmatched_requirements=unmatched,
    )
