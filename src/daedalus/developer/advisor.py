"""Deterministic, provider-neutral workflow advisor for building AI systems."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from daedalus.developer.models import (
    NON_WAIVABLE_STAGES,
    STAGE_ORDER,
    AdvisorTurn,
    BuildPlan,
    BuildStep,
    DeveloperSession,
    ExperienceMode,
    GateResult,
    GateState,
    ProjectBrief,
    ProjectEvidence,
    Question,
    Stage,
    TaskKind,
    ToolIntent,
    ToolKey,
    utc_now,
    validate_safe_text,
)

_STAGE_TEXT: dict[Stage, tuple[str, str]] = {
    Stage.DISCOVERY: (
        "Define the problem",
        "Turn the idea into a measurable user outcome and an explicit input/output contract.",
    ),
    Stage.RECOVERY: (
        "Inventory and protect recoverable work",
        "Verify the project, developer session, run journal, checkpoints, and last backup before changing more state.",
    ),
    Stage.DATA: (
        "Prove the data is usable",
        "Document source, permission, target, integrity, and a leakage-resistant split.",
    ),
    Stage.BASELINE: (
        "Establish a simple baseline",
        "Measure the simplest credible method before adding neural-network complexity.",
    ),
    Stage.ARCHITECTURE: (
        "Validate the model contract",
        "Choose shapes, objective, and metric that match the task and resource limits.",
    ),
    Stage.EXPERIMENT: (
        "Plan a reproducible experiment",
        "Fix seeds, budget, stop rules, tracked inputs, and expected artifacts before training.",
    ),
    Stage.EVALUATION: (
        "Evaluate outside the training loop",
        "Use held-out evidence, failure slices, and robustness checks to test the real claim.",
    ),
    Stage.DEPLOYMENT: (
        "Design operation and recovery",
        "Specify runtime limits, monitoring, rollback, and human ownership before release.",
    ),
    Stage.SECURITY: (
        "Close safety and supply-chain gaps",
        "Review privacy, threats, licensing, dependencies, secrets, and recoverable backup.",
    ),
    Stage.RELEASE: (
        "Assemble release evidence",
        "Publish only after the model card, acceptance decision, and Release Guard pass agree.",
    ),
}

_TASK_RECOMMENDATIONS: dict[TaskKind, dict[str, str]] = {
    TaskKind.CLASSIFICATION: {
        "baseline": "majority-class or logistic-regression baseline",
        "loss": "cross-entropy",
        "metric": "confusion matrix plus precision, recall, and F1",
    },
    TaskKind.REGRESSION: {
        "baseline": "mean predictor followed by linear regression",
        "loss": "mean-squared or mean-absolute error",
        "metric": "MAE, RMSE, and residual slices",
    },
    TaskKind.GENERATION: {
        "baseline": "frequency or n-gram generator",
        "loss": "next-token cross-entropy",
        "metric": "held-out loss plus a blinded quality and safety rubric",
    },
    TaskKind.RANKING: {
        "baseline": "fixed scoring heuristic",
        "loss": "pairwise or listwise ranking objective",
        "metric": "NDCG and MRR at a declared cutoff",
    },
    TaskKind.ANOMALY: {
        "baseline": "robust statistical threshold",
        "loss": "reconstruction or one-class objective",
        "metric": "precision-recall at the permitted alert budget",
    },
    TaskKind.CUSTOM: {
        "baseline": "documented rule-based comparator",
        "loss": "an objective derived from the output contract",
        "metric": "a held-out task-specific acceptance measure",
    },
}


def _q(
    question_id: str,
    stage: Stage,
    prompt: str,
    explanation: str,
    *,
    value_type: str = "text",
    example: str = "",
    choices: tuple[str, ...] = (),
    recommended: Any = None,
) -> Question:
    return Question(
        question_id,
        stage,
        prompt,
        explanation,
        True,
        value_type,
        example,
        choices,
        recommended,
    )


def _questions(task: TaskKind) -> tuple[Question, ...]:
    recommendation = _TASK_RECOMMENDATIONS[task]
    return (
        _q("outcome", Stage.DISCOVERY, "What useful outcome should this system create?", "Describe the change for a person or process, not a model name.", example="Flag likely equipment failures early enough for review."),
        _q("users", Stage.DISCOVERY, "Who uses or is affected by the output?", "Include the decision owner and anyone who could be harmed by an error.", example="Maintenance coordinators and equipment operators."),
        _q("inputs", Stage.DISCOVERY, "What information is available at prediction time?", "Training-only information cannot appear in the live input contract."),
        _q("outputs", Stage.DISCOVERY, "What exact output must the system return?", "Name the shape, units, classes, or response format."),
        _q("success_metric", Stage.DISCOVERY, "What measured result counts as success?", "Include a metric, population, threshold, and comparison where possible."),
        _q("recovery_owner", Stage.RECOVERY, "Who owns recovery decisions for this project?", "Name the accountable operator who can compare restored evidence and choose what becomes active."),
        _q("restore_destination", Stage.RECOVERY, "Which new, currently nonexistent directory may receive a restore?", "A restore must never overwrite the active workspace, source tree, backup root, or an existing path."),
        _q("recovery_drill_result", Stage.RECOVERY, "What was the result of the latest restore drill?", "Record when the backup was verified, where it was restored, and which hashes or checks passed."),
        _q("data_source", Stage.DATA, "Where will examples come from?", "Name the owner, collection process, and time range; do not paste credentials."),
        _q("data_rights_confirmed", Stage.DATA, "Are collection rights and permitted uses confirmed?", "A local file is not automatically licensed or consented for model training.", value_type="bool"),
        _q("target_definition", Stage.DATA, "How is the target or desired output defined?", "State who labels it, when it becomes known, and ambiguous cases."),
        _q("split_strategy", Stage.DATA, "How will train, validation, and test data be separated?", "Split by entity or time when random rows could leak related examples.", recommended="train/validation/test split grouped by the real deployment unit"),
        _q("baseline_choice", Stage.BASELINE, "Which simple baseline will you measure first?", "A baseline proves whether added complexity earns its cost.", recommended=recommendation["baseline"]),
        _q("baseline_success_threshold", Stage.BASELINE, "What must a learned model beat?", "Use the same held-out metric and decision threshold for a fair comparison.", recommended=recommendation["metric"]),
        _q("architecture_summary", Stage.ARCHITECTURE, "What model family and layer contract will you test?", "Start with the smallest capacity that can represent the relationship."),
        _q("loss_function", Stage.ARCHITECTURE, "Which training loss matches the output?", "The loss supplies gradients; it need not be the only reported metric.", recommended=recommendation["loss"]),
        _q("primary_metric", Stage.ARCHITECTURE, "Which primary evaluation metric matches the user cost?", "Accuracy alone can hide class, threshold, or slice failures.", recommended=recommendation["metric"]),
        _q("shape_contract", Stage.ARCHITECTURE, "Write the input, batch, hidden, and output shapes.", "Validate every adjacent dimension before allocating memory."),
        _q("seed", Stage.EXPERIMENT, "Which reproducibility seed will this run use?", "A seed is evidence only when data, code, configuration, and versions are recorded too.", value_type="int", recommended=47),
        _q("training_budget", Stage.EXPERIMENT, "What is the bounded training budget?", "State epochs or steps plus time and memory limits."),
        _q("stop_rule", Stage.EXPERIMENT, "What stops training or rejects the run?", "Examples include validation patience, non-finite values, and a fixed compute ceiling."),
        _q("holdout_result", Stage.EVALUATION, "What is the untouched hold-out result?", "Report the declared primary metric and uncertainty where feasible."),
        _q("failure_analysis", Stage.EVALUATION, "Which failures and user-relevant slices were inspected?", "Keep representative false positives, false negatives, or poor outputs as evidence."),
        _q("robustness_checks", Stage.EVALUATION, "Which shifts, edge cases, or stress tests passed?", "Test realistic missingness, scale, drift, and invalid-input behavior."),
        _q("deployment_target", Stage.DEPLOYMENT, "Where and how will inference run?", "Name the device or service boundary and whether a human reviews the output."),
        _q("latency_budget", Stage.DEPLOYMENT, "What are the latency and memory budgets?", "Use an explicit percentile and device class rather than 'fast'."),
        _q("monitoring_plan", Stage.DEPLOYMENT, "What will detect drift or operational failure?", "Define owners, alert thresholds, data retention, and review cadence."),
        _q("rollback_plan", Stage.DEPLOYMENT, "How is this version disabled or rolled back?", "A rollback needs a known-good artifact and an accountable operator."),
        _q("privacy_review", Stage.SECURITY, "What private or regulated information is handled?", "Describe minimization, access, retention, deletion, and logging controls."),
        _q("threats", Stage.SECURITY, "Which misuse and technical threats matter here?", "Consider untrusted data, poisoned artifacts, unsafe output use, secrets, and dependencies."),
        _q("licenses_reviewed", Stage.SECURITY, "Have data, code, model, and dependency licenses been reviewed?", "Release must respect every incorporated artifact's terms.", value_type="bool"),
        _q("release_version", Stage.RELEASE, "Which immutable release version is being proposed?", "Use a version that maps to code, configuration, data identity, and artifact hashes."),
        _q("release_acceptance", Stage.RELEASE, "Does an accountable reviewer accept the recorded limitations?", "Acceptance is an explicit human decision, never an automatic bot action.", value_type="bool"),
    )


_REQUIRED_ANSWERS: dict[Stage, tuple[str, ...]] = {
    stage: tuple(question.id for question in _questions(TaskKind.CUSTOM) if question.stage == stage)
    for stage in STAGE_ORDER
}

_EVIDENCE_GATES: dict[Stage, tuple[tuple[str, str], ...]] = {
    Stage.DISCOVERY: (),
    Stage.RECOVERY: (
        ("session_inventory_complete", "developer-session inventory is incomplete"),
        ("run_inventory_complete", "training-run inventory is incomplete"),
        ("checkpoint_inventory_complete", "checkpoint inventory is incomplete"),
        ("backup_current", "no current verified backup is confirmed"),
        ("restore_target_safe", "restore target is absent or not proven safe and non-overwriting"),
    ),
    Stage.DATA: (
        ("workspace_ready", "private workspace ownership is not verified"),
        ("project_manifest_valid", "project manifest is missing or invalid"),
        ("dataset_present", "no dataset is registered"),
        ("dataset_integrity", "dataset integrity is not verified"),
    ),
    Stage.BASELINE: (("baseline_recorded", "baseline result is not recorded"),),
    Stage.ARCHITECTURE: (("architecture_validated", "architecture shapes are not validated"),),
    Stage.EXPERIMENT: (("experiment_plan_present", "experiment plan artifact is missing"),),
    Stage.EVALUATION: (
        ("run_completed", "no completed run is recorded"),
        ("heldout_metrics_present", "held-out metrics are not recorded"),
    ),
    Stage.DEPLOYMENT: (("deployment_plan_present", "deployment runbook is missing"),),
    Stage.SECURITY: (
        ("threat_model_present", "threat model is missing"),
        ("secret_scan_passed", "secret scan has not passed"),
        ("dependency_scan_passed", "dependency scan has not passed"),
        ("backup_current", "current verified backup is not confirmed"),
    ),
    Stage.RELEASE: (
        ("model_card_present", "model card is missing"),
        ("release_guard_passed", "Release Guard has not passed the exact release content"),
    ),
}

def _tool_intents(stage: Stage, task: TaskKind) -> tuple[ToolIntent, ...]:
    recommendations = _TASK_RECOMMENDATIONS[task]
    mapping: dict[Stage, tuple[ToolIntent, ...]] = {
        Stage.DISCOVERY: (
            ToolIntent(ToolKey.LEARN, "Open Learning Atlas", "Review the task and metric concepts before committing to an architecture."),
        ),
        Stage.RECOVERY: (
            ToolIntent(
                ToolKey.VAULT,
                "Inventory and verify recovery",
                "Use the verified backup and restore only into a new, empty destination.",
                {"restore_mode": "new-directory-only"},
            ),
        ),
        Stage.DATA: (
            ToolIntent(ToolKey.TRAINING, "Import and inspect data", "Use validated private dataset ingestion and preserve its checksum."),
        ),
        Stage.BASELINE: (
            ToolIntent(ToolKey.WORKSHOP, "Implement the baseline", f"Start with a {recommendations['baseline']} and record the held-out result."),
        ),
        Stage.ARCHITECTURE: (
            ToolIntent(ToolKey.ARCHITECTURE, "Validate shapes", "Prove every adjacent tensor dimension before training."),
            ToolIntent(ToolKey.CALCULATOR, "Estimate resources", "Check parameters, activation memory, and training budget."),
        ),
        Stage.EXPERIMENT: (
            ToolIntent(ToolKey.TRAINING, "Run a bounded experiment", "Use the recorded seed, stop rule, and budget."),
        ),
        Stage.EVALUATION: (
            ToolIntent(ToolKey.EVALUATE, "Inspect held-out evidence", "Evaluate a trusted checkpoint without changing the training split."),
        ),
        Stage.DEPLOYMENT: (
            ToolIntent(ToolKey.CALCULATOR, "Check inference limits", "Compare model memory and latency evidence with the target device budget."),
            ToolIntent(ToolKey.WORKSHOP, "Build the integration", "Keep packaging code in the private project and test rollback behavior."),
        ),
        Stage.SECURITY: (
            ToolIntent(ToolKey.VAULT, "Verify backup", "Confirm a current, restorable copy before release work."),
            ToolIntent(ToolKey.GUARD, "Run security checks", "Scan secrets, dependencies, and exact outgoing content."),
        ),
        Stage.RELEASE: (
            ToolIntent(ToolKey.GUARD, "Run final Release Guard", "Publish only the exact reviewed content when every blocking check passes."),
        ),
    }
    return mapping[stage]


def _has_answer(answers: dict[str, Any], question: Question) -> bool:
    if question.id not in answers:
        return False
    value = answers[question.id]
    if question.value_type == "bool":
        return value is True
    if question.value_type == "int":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    return isinstance(value, str) and bool(value.strip())


class DeveloperAdvisor:
    """Pure deterministic advisor with visible, stage-gated decisions."""

    def start(
        self,
        project_root,
        brief: ProjectBrief,
        mode: ExperienceMode = ExperienceMode.BEGINNER,
        *,
        session_id: str | None = None,
    ) -> DeveloperSession:
        return DeveloperSession.create(project_root, brief, mode, session_id=session_id)

    def questions(self, session: DeveloperSession, stage: Stage | None = None) -> tuple[Question, ...]:
        all_questions = _questions(session.brief.task_kind)
        if stage is None:
            return all_questions
        selected = tuple(question for question in all_questions if question.stage == Stage(stage))
        return selected

    def answer(self, session: DeveloperSession, question_id: str, value: Any) -> DeveloperSession:
        question = next((item for item in self.questions(session) if item.id == question_id), None)
        if question is None:
            raise KeyError(f"unknown developer question: {question_id}")
        if question.value_type == "bool":
            if not isinstance(value, bool):
                raise TypeError(f"{question_id} requires a yes/no value")
            clean: Any = value
        elif question.value_type == "int":
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 2_147_483_647:
                raise ValueError(f"{question_id} requires an integer from 0 to 2147483647")
            clean = value
        else:
            clean = validate_safe_text(value, field_name=question_id)
            if question.choices and clean not in question.choices:
                raise ValueError(f"{question_id} must be one of the documented choices")
        return session.with_answer(question_id, clean)

    def waive(self, session: DeveloperSession, stage: Stage, reason: str) -> DeveloperSession:
        resolved = Stage(stage)
        if resolved in NON_WAIVABLE_STAGES:
            raise ValueError(f"the {resolved.value} gate cannot be waived")
        clean = validate_safe_text(reason, field_name="waiver reason")
        if len(clean) < 12:
            raise ValueError("waiver reason must explain the tradeoff in at least 12 characters")
        return session.with_waiver(resolved, clean)

    def assess(
        self,
        session: DeveloperSession,
        evidence: ProjectEvidence | None = None,
    ) -> tuple[GateResult, ...]:
        observed = evidence or ProjectEvidence()
        questions = self.questions(session)
        answer_map = dict(session.answers)
        results: list[GateResult] = []
        for stage in STAGE_ORDER:
            missing: list[str] = []
            stage_questions = {item.id: item for item in questions if item.stage == stage}
            for question_id in _REQUIRED_ANSWERS[stage]:
                question = stage_questions[question_id]
                if not _has_answer(answer_map, question):
                    missing.append(question.prompt)
            for attribute, message in _EVIDENCE_GATES[stage]:
                if getattr(observed, attribute) is not True:
                    missing.append(message)

            reasons = [
                _STAGE_TEXT[stage][1],
                f"Task-specific guidance uses the {session.brief.task_kind.value} workflow.",
            ]
            # Session validation already rejects protected waiver keys. Keep
            # assessment fail-closed if an impossible instance is injected.
            waiver = (
                None
                if stage in NON_WAIVABLE_STAGES
                else session.waivers.get(stage.value)
            )
            if waiver:
                state = GateState.WAIVED
                reasons.append("An accountable user explicitly recorded a waiver; missing evidence remains visible.")
            elif missing:
                state = GateState.BLOCKED
                reasons.append(f"{len(missing)} required item(s) remain unresolved.")
            else:
                state = GateState.PASSED
                reasons.append("All required answers and observable evidence for this stage are present.")
            results.append(GateResult(stage, state, tuple(reasons), tuple(missing), waiver))
        return tuple(results)

    def current_stage(
        self,
        session: DeveloperSession,
        evidence: ProjectEvidence | None = None,
    ) -> Stage:
        results = self.assess(session, evidence)
        for result in results:
            if result.state not in {GateState.PASSED, GateState.WAIVED}:
                return result.stage
        return Stage.RELEASE

    def next_turn(
        self,
        session: DeveloperSession,
        evidence: ProjectEvidence | None = None,
    ) -> AdvisorTurn:
        stage = self.current_stage(session, evidence)
        gate = next(result for result in self.assess(session, evidence) if result.stage == stage)
        unanswered = tuple(
            question
            for question in self.questions(session, stage)
            if not _has_answer(dict(session.answers), question)
        )
        if session.mode == ExperienceMode.BEGINNER:
            visible = unanswered[:1]
            summary = (
                "Work through one decision at a time. The bot will explain why the evidence matters "
                "and will not run code or publish anything for you."
            )
        elif session.mode == ExperienceMode.BUILDER:
            visible = unanswered
            summary = "Complete this stage as a compact engineering checklist, then verify its evidence."
        else:
            visible = unanswered
            summary = "Review the gate matrix, supply canonical fields, and record any permitted waiver rationale."
        title, _objective = _STAGE_TEXT[stage]
        return AdvisorTurn(
            stage,
            title,
            summary,
            gate.reasons,
            visible,
            gate,
            _tool_intents(stage, session.brief.task_kind),
        )

    def build_plan(
        self,
        session: DeveloperSession,
        evidence: ProjectEvidence | None = None,
    ) -> BuildPlan:
        gates = {result.stage: result for result in self.assess(session, evidence)}
        steps = tuple(
            BuildStep(
                stage,
                _STAGE_TEXT[stage][0],
                _STAGE_TEXT[stage][1],
                gates[stage],
                _tool_intents(stage, session.brief.task_kind),
            )
            for stage in STAGE_ORDER
        )
        return BuildPlan(
            session.id,
            session.mode,
            session.brief.task_kind,
            steps,
            session.updated_utc,
        )

    def change_mode(
        self, session: DeveloperSession, mode: ExperienceMode
    ) -> DeveloperSession:
        return replace(session, mode=ExperienceMode(mode), updated_utc=utc_now())


__all__ = ["DeveloperAdvisor"]
