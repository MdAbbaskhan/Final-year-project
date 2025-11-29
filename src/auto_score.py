import pandas as pd
from pathlib import Path

# Try auto-detect
candidates = [
    "mongo_eval_dump.csv",
    "rag_eval_dump.csv",
    "llm_rag_3x3_results.csv",
    "rag_results.csv"
]

INPUT = None
for c in candidates:
    if Path(c).exists():
        INPUT = c
        break

if INPUT is None:
    raise FileNotFoundError(
        "\n❌ No evaluation CSV found.\n"
        "Make sure you exported Mongo results first.\n"
        "Look for one of these:\n- mongo_eval_dump.csv\n- llm_rag_3x3_results.csv\n- rag_eval_dump.csv\n"
    )

print(f"📁 Loading: {INPUT}")
df = pd.read_csv(INPUT)

def auto_score(answer: str, question: str):
    if not isinstance(answer, str):
        return (0,0,0)

    ans = answer.lower()

    # no answer
    if "i don't know" in ans or "timeout" in ans:
        return (0,0,0)

    # hallucinations
    if "pride and prejudice" in ans or "beach" in ans or "old man" in ans:
        return (1,0,1)

    ## characters
    if "who are the main characters" in question.lower():
        names = ["alice","gryphon","white rabbit","king","queen","knave"]
        hits = sum([n in ans for n in names])

        if hits >= 5: return (4,4,3)
        if hits >= 3: return (3,2,2)
        if hits >= 1: return (2,1,1)
        return (1,0,1)

    ## main idea
    if "main idea" in question.lower():
        if "not" in ans or "cannot" in ans:
            return (0,0,0)
        if "relationship" in ans or "bond" in ans:
            return (3,3,2)
        return (2,2,1)

    ## describe documents
    if "document" in question.lower():
        if "verses" in ans or "accident" in ans:
            return (4,4,3)
        if "paper" in ans or "letter" in ans:
            return (3,3,2)
        return (1,1,1)

    return (1,1,1)


# apply to rows
scores = df.apply(lambda r: auto_score(r["answer"], r["question"]), axis=1)

df["relevance_score"] = [s[0] for s in scores]
df["correctness_score"] = [s[1] for s in scores]
df["completeness_score"] = [s[2] for s in scores]

out = f"{INPUT.replace('.csv','')}_scored.csv"
df.to_csv(out, index=False)

print(f"\n✅ DONE — Saved scored file 👉 {out}")
