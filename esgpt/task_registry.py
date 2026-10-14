from typing import Dict, Optional

# Public-release task taxonomy aligned with the ESG-Instruct and ESGU-Bench
# composition tables used in the paper. Internal construction-only tasks are
# intentionally omitted from this registry.
CORE_TASK_GROUPS: Dict[str, set[str]] = {
    "general_instruction_chat": {
        "alpaca_general_instruction",
        "alpaca_instruction_following",
    },
    "esg_relevance_detection": {
        "climate_detection_classification",
        "environmental_claims_classification",
        "ftse_esg_relevance_classification",
        "korean_esg_relevance_classification",
    },
    "esg_topic_identification": {
        "aspect_extraction",
        "environmental_classification",
        "esg_activity_concept_matching",
        "esg_category_classification",
        "esg_topic_classification",
        "ftse_esg_primary_category_classification",
        "governance_classification",
        "korean_esg_category_classification",
        "mlesg1_esg_issue_classification",
        "sasb_issue_topic_classification",
        "sasb_parent_category_classification",
        "social_classification",
        "tcfd_recommendations_classification",
    },
    "esg_commitment_analysis": {
        "action_classification",
        "climate_commitments_actions_classification",
        "climate_specificity_classification",
        "esg_action_classification",
        "netzero_reduction_target_classification",
        "promise_status_classification",
        "promise_verification_timeline_classification",
    },
    "esg_impact_analysis": {
        "climate_sentiment_classification",
        "esg_aspect_sentiment_classification",
        "impact_duration_classification",
        "impact_type_classification",
        "mlesg2_impact_type_classification",
        "mlesg2_japanese_sentiment_classification",
        "mlesg3_impact_duration_classification",
        "mlesg3_impact_level_classification",
        "mlesg3_impact_type_classification",
    },
    "esg_claim_verification": {
        "claim_verification",
        "promise_evidence_quality_classification",
        "promise_evidence_status_classification",
    },
    "esg_qa": {
        "esg_qa",
        "sustainableqa_factoid_answering",
        "sustainableqa_non_factoid_answering",
        "sustainableqa_table_answering",
    },
}

CORE_TASK_GROUP_DISPLAY_NAMES: Dict[str, str] = {
    "general_instruction_chat": "General",
    "esg_relevance_detection": "ESG Relevance Detection",
    "esg_topic_identification": "ESG Topic Identification",
    "esg_commitment_analysis": "ESG Commitment Analysis",
    "esg_impact_analysis": "ESG Impact Analysis",
    "esg_claim_verification": "ESG Claim Verification",
    "esg_qa": "ESG Question Answering",
}

EXTERNAL_EVAL_DISPLAY_NAMES: Dict[str, str] = {
    "esgenius__esg_qa": "ESG Reasoning - ESGenius",
}

EXTERNAL_BENCHMARK_ORDER = (
    "esgenius__esg_qa",
)

DEFAULT_EXTERNAL_BENCHMARKS = EXTERNAL_BENCHMARK_ORDER

CORE_TASK_GROUP_ALIASES: Dict[str, str] = {
    "general": "general_instruction_chat",
    "general_instruction": "general_instruction_chat",
    "general_instruction_following": "general_instruction_chat",
    "instruction_following": "general_instruction_chat",
    "General": "general_instruction_chat",
    "General Instruction Following": "general_instruction_chat",
    "relevance_detection": "esg_relevance_detection",
    "esg_topic_taxonomy_classification": "esg_topic_identification",
    "commitment_target_disclosure_classification": "esg_commitment_analysis",
    "sentiment_impact_classification": "esg_impact_analysis",
    "claim_promise_evidence_verification": "esg_claim_verification",
    "context_grounded_qa": "esg_qa",
    "question_answering": "esg_qa",
    "esg_reasoning": "esg_qa",
    "ESG Relevance Detection": "esg_relevance_detection",
    "ESG Topic Identification": "esg_topic_identification",
    "ESG Commitment Analysis": "esg_commitment_analysis",
    "ESG Impact Analysis": "esg_impact_analysis",
    "ESG Claim Verification": "esg_claim_verification",
    "ESG QA": "esg_qa",
    "ESG Question Answering": "esg_qa",
}

TASK_TO_CORE_GROUP = {
    task_name: group_name
    for group_name, task_names in CORE_TASK_GROUPS.items()
    for task_name in task_names
}


def task_name_from_detail_folder(detail_task_folder: str) -> str:
    if "__" not in detail_task_folder:
        return detail_task_folder
    return detail_task_folder.split("__", 1)[1]


def normalize_core_task_group_name(group_name: str) -> str:
    return CORE_TASK_GROUP_ALIASES.get(group_name, group_name)


def core_task_group_display_name(group_name: str) -> str:
    normalized = normalize_core_task_group_name(group_name)
    return CORE_TASK_GROUP_DISPLAY_NAMES.get(normalized, normalized)


def infer_core_task_group(detail_task_folder: str, task_name: Optional[str] = None) -> str:
    resolved_task_name = task_name or task_name_from_detail_folder(detail_task_folder)
    return TASK_TO_CORE_GROUP.get(resolved_task_name, "unmapped")
