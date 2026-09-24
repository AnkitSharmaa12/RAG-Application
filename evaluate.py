"""Golden-question check. Run after build_index.py:  python evaluate.py

Each case: (question, [acceptable substrings, any one must appear], should_refuse).
Also prints top retrieval scores so MIN_SCORE in rag/config.py can be tuned.
"""
import sys
import time

from rag.pipeline import NOT_FOUND_MESSAGE, RagPipeline

CASES = [
    ("What is MecroTech's phone number?", ["70118 48978", "7011848978"], False),
    ("What is the email address of MecroTech?", ["hello@mecro.tech"], False),
    ("Where is MecroTech located?", ["najafgarh", "new delhi"], False),
    ("Who founded MecroTech?", ["nitish gulia"], False),
    ("Who is the COO?", ["megha gupta"], False),
    ("How much commission does the referral program pay?", ["5-10%", "5–10%", "5 to 10"], False),
    ("How quickly do referrers get paid?", ["24-48", "24–48"], False),
    ("How long does it take to build an MVP?", ["2-4 weeks", "2–4 weeks"], False),
    ("How much cheaper is MecroTech than a traditional agency?", ["50-70%", "50–70%"], False),
    ("Which courts have jurisdiction over disputes?", ["new delhi"], False),
    ("How long do refunds take to process?", ["7-14", "7–14"], False),
    ("Who owns the source code after the project?", ["full and final payment", "client"], False),
    ("What tech stack do you use?", ["next.js", "react"], False),
    ("Is an NDA available?", ["nda", "yes"], False),
    ("Which analytics tool does the website use?", ["clarity"], False),
    ("What are the business hours?", ["09:00", "9", "monday"], False),
    # follow-up handled separately below
    ("Who is the CEO of Google?", [], True),
    ("What is the monthly price of your basic plan in dollars?", ["don't", "not", "no price", "quote", "publish"], False),
    ("Give me a recipe for pasta.", [], True),
]

FOLLOW_UP = [("What is the refund policy?", ["refund"]), ("And how long does it take to process them?", ["7-14", "7–14"])]


def main() -> int:
    rag = RagPipeline()
    failures = 0
    for q, expected, should_refuse in CASES:
        answer, sources = rag.answer(q)
        top = f"{sources[0].score:.2f}" if sources else "none"
        low = answer.lower()
        if should_refuse:
            ok = answer == NOT_FOUND_MESSAGE or any(
                p in low for p in ("only help", "don't have", "do not have", "couldn't find", "not contain", "unrelated")
            )
        else:
            ok = any(e.lower() in low for e in expected)
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] top={top} | {q}")
        if not ok:
            print("        ->", answer.replace("\n", " ")[:300])
        time.sleep(0.5)

    history = []
    for q, expected in FOLLOW_UP:
        answer, _ = rag.answer(q, history)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
    ok = any(e in answer.lower() for e in FOLLOW_UP[-1][1])
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] follow-up | {FOLLOW_UP[-1][0]}")
    if not ok:
        print("        ->", answer.replace("\n", " ")[:300])

    print(f"\n{len(CASES) + 1 - failures}/{len(CASES) + 1} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
