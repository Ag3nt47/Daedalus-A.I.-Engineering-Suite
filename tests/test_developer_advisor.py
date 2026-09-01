from __future__ import annotations

import json
from dataclasses import fields, replace

import pytest

from daedalus.developer import (
    NON_WAIVABLE_STAGES,
    STAGE_ORDER,
    DeveloperAdvisor,
    ExperienceMode,
    GateState,
    ProjectBrief,
    ProjectEvidence,
    Stage,
    TaskKind,
    session_from_json,
    session_to_json,
)


def brief(task: TaskKind = TaskKind.CLASSIFICATION) -> ProjectBrief:
    return ProjectBrief(
        "Bearing Watch",
        "Flag likely bearing failures before an outage",
        "Maintenance coordinators and equipment operators",
        task,
        "A window of numeric vibration features",
        "A reviewed risk class and confidence",
        "Recall of at least 0.90 at no more than five alerts per day",
        ("Runs offline", "Fits in 256 MiB"),
    )


def all_evidence() -> ProjectEvidence:
    return ProjectEvidence(**{item.name: True for item in fields(ProjectEvidence)})


def answer_all(advisor: DeveloperAdvisor, session):
    for question in advisor.questions(session):
        if question.id in session.answers:
            continue
        if question.value_type == "bool":
            value = True
        elif question.value_type == "int":
            value = 47
        else:
            value = question.recommended_answer or f"Recorded response for {question.id}"
        session = advisor.answer(session, question.id, value)
    return session


def test_ten_stage_order_and_recovery_gate(tmp_path) -> None:
    advisor = DeveloperAdvisor()
    project = tmp_path / "workspace" / "projects" / "bearing"
    session = advisor.start(project, brief())

    assert len(STAGE_ORDER) == 10
    assert STAGE_ORDER[:3] == (Stage.DISCOVERY, Stage.RECOVERY, Stage.DATA)
    results = advisor.assess(session)
    assert results[0].state == GateState.PASSED
    assert results[1].state == GateState.BLOCKED
    assert "backup" in " ".join(results[1].missing).casefold()
    assert advisor.current_stage(session) == Stage.RECOVERY


def test_modes_change_presentation_not_gate_logic(tmp_path) -> None:
    advisor = DeveloperAdvisor()
    session = advisor.start(tmp_path / "project", brief(), ExperienceMode.BEGINNER)
    beginner = advisor.next_turn(session)
    builder_session = advisor.change_mode(session, ExperienceMode.BUILDER)
    expert_session = advisor.change_mode(session, ExperienceMode.EXPERT)

    assert beginner.stage == Stage.RECOVERY
    assert len(beginner.questions) == 1
    assert len(advisor.next_turn(builder_session).questions) == 3
    assert len(advisor.next_turn(expert_session).questions) == 3
    assert [result.state for result in advisor.assess(session)] == [
        result.state for result in advisor.assess(builder_session)
    ]


@pytest.mark.parametrize("task", tuple(TaskKind))
def test_every_task_has_visible_recommendations(tmp_path, task: TaskKind) -> None:
    advisor = DeveloperAdvisor()
    session = advisor.start(tmp_path / task.value, brief(task))
    baseline = next(question for question in advisor.questions(session) if question.id == "baseline_choice")
    loss = next(question for question in advisor.questions(session) if question.id == "loss_function")
    metric = next(question for question in advisor.questions(session) if question.id == "primary_metric")

    assert baseline.recommended_answer
    assert loss.recommended_answer
    assert metric.recommended_answer
    assert task.value in advisor.assess(session)[0].reasons[1]


def test_all_answers_and_evidence_pass_all_gates(tmp_path) -> None:
    advisor = DeveloperAdvisor()
    session = answer_all(advisor, advisor.start(tmp_path / "project", brief()))
    results = advisor.assess(session, all_evidence())
    plan = advisor.build_plan(session, all_evidence())

    assert all(result.state == GateState.PASSED for result in results)
    assert len(plan.steps) == len(STAGE_ORDER)
    assert tuple(step.stage for step in plan.steps) == STAGE_ORDER
    assert all(step.tool_intents for step in plan.steps)
    assert plan.generated_utc == session.updated_utc


def test_answer_types_unknown_fields_and_credentials_are_rejected(tmp_path) -> None:
    advisor = DeveloperAdvisor()
    session = advisor.start(tmp_path / "project", brief())

    with pytest.raises(KeyError):
        advisor.answer(session, "made_up", "value")
    with pytest.raises(TypeError):
        advisor.answer(session, "data_rights_confirmed", "yes")
    with pytest.raises(ValueError):
        advisor.answer(session, "seed", -1)
    credential = "".join(("ghp", "_", "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"))
    with pytest.raises(ValueError) as captured:
        advisor.answer(session, "data_source", credential)
    assert credential not in str(captured.value)


def test_waivers_are_explicit_and_protected_gates_cannot_be_waived(tmp_path) -> None:
    advisor = DeveloperAdvisor()
    session = advisor.start(tmp_path / "project", brief())
    waived = advisor.waive(session, Stage.BASELINE, "Prototype lacks a baseline runner today")
    baseline = advisor.assess(waived)[STAGE_ORDER.index(Stage.BASELINE)]

    assert baseline.state == GateState.WAIVED
    assert baseline.missing
    assert baseline.waiver_reason
    for protected in (
        Stage.DISCOVERY,
        Stage.RECOVERY,
        Stage.DATA,
        Stage.SECURITY,
        Stage.RELEASE,
    ):
        with pytest.raises(ValueError):
            advisor.waive(session, protected, "A long but unacceptable waiver reason")


@pytest.mark.parametrize("protected", tuple(NON_WAIVABLE_STAGES))
def test_protected_waivers_are_rejected_by_model_and_import(tmp_path, protected: Stage) -> None:
    advisor = DeveloperAdvisor()
    session = advisor.start(tmp_path / "project", brief())
    reason = "Crafted but unacceptable protected gate waiver"

    with pytest.raises(ValueError):
        session.with_waiver(protected, reason)
    with pytest.raises(ValueError):
        replace(session, waivers={protected.value: reason})

    raw = json.loads(session_to_json(session))
    raw["waivers"] = {protected.value: reason}
    with pytest.raises(ValueError):
        session_from_json(json.dumps(raw))


def test_assessment_defensively_ignores_impossible_protected_waiver(tmp_path) -> None:
    advisor = DeveloperAdvisor()
    session = advisor.start(tmp_path / "project", brief())
    object.__setattr__(
        session,
        "waivers",
        {Stage.RELEASE.value: "Injected impossible release waiver state"},
    )

    release = advisor.assess(session)[STAGE_ORDER.index(Stage.RELEASE)]

    assert release.state == GateState.BLOCKED
    assert release.waiver_reason is None
    assert release.missing


def test_recovery_tool_intent_is_new_directory_only(tmp_path) -> None:
    advisor = DeveloperAdvisor()
    session = advisor.start(tmp_path / "project", brief())
    turn = advisor.next_turn(session)

    assert turn.stage == Stage.RECOVERY
    assert turn.tool_intents[0].tool_key.value == "vault"
    assert turn.tool_intents[0].payload["restore_mode"] == "new-directory-only"
