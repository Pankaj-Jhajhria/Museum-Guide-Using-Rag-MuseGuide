"""Run QA verification across 5 realistic questions testing artwork-specific, thematic, and unsupported queries."""

import json
import re
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.rag.answer import AnswerGenerator


def run_qa_tests():
    gen = AnswerGenerator()
    eval_file = Path("tests/eval_questions.json")
    with open(eval_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print("\n" + "="*70)
    print("MUSEGUIDE QA VERIFICATION: 5 REALISTIC QUESTIONS")
    print("="*70)

    for idx, case in enumerate(cases, 1):
        q = case["question"]
        cat = case["category"]
        expected_grounded = case["should_be_grounded"]

        resp = gen.generate(q)

        # Check inline citation markers vs structured citations
        text_markers = set(re.findall(r"\[(\d+)\]", resp.answer))
        cit_ids = {str(c.citation_id) for c in resp.citations}
        markers_match = (text_markers == cit_ids)

        print(f"\n[TEST {idx}/5] Category: {cat}")
        print(f"QUESTION: {q}")
        print(f"GROUNDED: {resp.grounded} (Expected: {expected_grounded})")
        print(f"CITATIONS RETURNED: {len(resp.citations)}")
        print(f"MARKERS IN TEXT MATCH CITATION OBJECTS 1:1: {markers_match} ({cit_ids})")
        print("\nGENERATED ANSWER:\n" + resp.answer)

        if resp.citations:
            print("\nSTRUCTURED CITATIONS:")
            for cit in resp.citations:
                print(f"  [{cit.citation_id}] \"{cit.title}\" | {cit.creator or 'Unknown'} | {cit.creation_date or 'N/A'}")
                print(f"       Culture: {cit.culture or 'N/A'} | Dept: {cit.department or 'N/A'}")
                print(f"       Score: {cit.similarity_score:.4f} | Link: {cit.museum_url}")
        print("-" * 70)


if __name__ == "__main__":
    run_qa_tests()
