from skills_ml.algorithms.skill_extractors.base import CandidateSkill
from skills_ml.algorithms.skill_extractors.tf import tf_sequence_from_candidate_skills

job_desc = "Hi we are hiring a data scientist. We would like somebody trained in Microsoft Office to do some pivot tables. Also they need to be skilled at coffeemaking. We pay a bunch of money"

candidate_skills = [
    CandidateSkill(
        skill_name="Microsoft Office",
        matched_skill_identifier=None,
        context="We would like somebody trained in Microsoft Office to do some pivot tables",
        start_index=69,
        confidence=1.0,
        document_id="123",
        document_type="JobPosting",
        source_object={"description": job_desc},
        skill_extractor_name="my_fake_skill_extractor"
    ),
    CandidateSkill(
        skill_name="coffeemaking",
        matched_skill_identifier=None,
        context="Also they need to be skilled at coffeemaking",
        start_index=143,
        confidence=1.0,
        document_id="123",
        document_type="JobPosting",
        source_object={"description": job_desc},
        skill_extractor_name="my_fake_skill_extractor"
    )
]

assert tf_sequence_from_candidate_skills(candidate_skills) == {
    'words': [
        ["Hi", "we", "are", "hiring", "a", "data", "scientist"],
        ["We", "would", "like", "somebody", "trained", "in", "Microsoft", "Office", "to", "do", "some", "pivot", "tables"],
        ["Also", "they", "need", "to", "be", "skilled", "at", "coffeemaking"],
        ["We", "pay", "a", "bunch", "of", "money"]
    ],
    'tags': [
        ["O", "O", "O", "O", "O", "O", "O"],
        ["O", "O", "O", "O", "O", "O", "B-SKILL" "I-SKILL", "O", "O", "O", "O", "O"],
        ["O", "O", "O", "O", "O", "O", "O", "S-SKILL"],
        ["O", "O", "O", "O", "O", "O"],
    ]
}
