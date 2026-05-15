import json
import os
import re
import time
from time import sleep

import pandas as pd
from datasets import load_from_disk
from dotenv import load_dotenv
from google import genai
from google.genai import types

from utils import score_submission

load_dotenv()

SPLIT = "validation"
SPLITS_PATH = "challenge-dataset"
OUT_PATH = f"output/submission_llm_text_{SPLIT}.csv"

MODEL = "gemini-3.1-pro-preview"

TEXT_METRICS = ["TCC", "MS", "M"]
VIDEO_METRICS = ["VC", "VA", "VE"]
ALL_METRICS = TEXT_METRICS + VIDEO_METRICS

# Placeholder video scores.
# Video metrics are 0-indexed: 0, 1, 2, 3.
DEFAULT_VIDEO_SCORE = 0

RUBRIC = """You are an expert evaluator of pedagogical analogies for Computer Science concepts.
Given a target concept, its description, and an analogy, score the analogy on three metrics.
Use these exact 0-indexed scales:

TCC - Target Concept Coverage: how well the analogy covers the topics in the description.
  0 = covers none, 1 = covers some, 2 = covers all

MS - Mapping Strength: logical soundness and consistency of source<->target correspondence.
  0 = weak, 1 = moderately strong, 2 = very strong

M - Metaphoricity: conceptual distance between source and target.
  0 = low (source is similar to target), 1 = moderate, 2 = high (source is very different from target)

Respond with ONLY a JSON object, no prose, no code fences:
{"TCC": <int>, "MS": <int>, "M": <int>}
"""


def build_prompt(target, description, analogy):
    return f"""{RUBRIC}

TARGET CONCEPT: {target}

DESCRIPTION: {description}

ANALOGY:
{analogy}

Scores (JSON only):"""


def parse_scores(text):
    # Strip code fences if the model ignores instructions.
    text = re.sub(r"```(?:json)?|```", "", text).strip()

    # Find the first {...} block.
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in: {text!r}")

    obj = json.loads(match.group(0))

    scores = {}
    for metric in TEXT_METRICS:
        if metric not in obj:
            raise ValueError(f"Missing metric {metric} in model output: {obj}")

        value = int(obj[metric])

        if value not in [0, 1, 2]:
            raise ValueError(
                f"Invalid value for {metric}: {value}. "
                "Allowed text values are 0, 1, 2."
            )

        scores[metric] = value

    return scores


def score_one(client, target, description, analogy, max_retries=3):
    prompt = build_prompt(target, description, analogy)

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(
                        thinking_level="MEDIUM",
                    ),
                ),
            )
            return parse_scores(resp.text)

        except Exception as e:
            print(f"    attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)

    # Fallback: middle of the text scale.
    return {m: 1 for m in TEXT_METRICS}


def make_submission_row(example_id, text_scores):
    return {
        "id": example_id,
        "TCC": text_scores["TCC"],
        "MS": text_scores["MS"],
        "M": text_scores["M"],
        "VC": DEFAULT_VIDEO_SCORE,
        "VA": DEFAULT_VIDEO_SCORE,
        "VE": DEFAULT_VIDEO_SCORE,
    }


def main():
    os.makedirs("output", exist_ok=True)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    dataset = load_from_disk(SPLITS_PATH)
    test = dataset[SPLIT]

    rows = []

    for i in range(len(test)):
        row = test[i]

        print(f"[{i + 1}/{len(test)}] {row['target']}")

        text_scores = score_one(
            client,
            row["target"],
            row["description"],
            row["analogy"],
        )

        submission_row = make_submission_row(i, text_scores)

        print(f"    -> {submission_row}")

        rows.append(submission_row)

        sleep(1)

    submission_df = pd.DataFrame(rows)

    # Ensure exact competition column order.
    submission_df = submission_df[["id", "TCC", "MS", "M", "VC", "VA", "VE"]]

    submission_df.to_csv(OUT_PATH, index=False)

    print(f"\nWrote CSV submission: {OUT_PATH}\n")
    print(submission_df.head())

    # Optional: score locally using the generated CSV.
    try:
        results = score_submission(OUT_PATH, SPLIT)

        print(f"\n{'metric':<10} {'tau-b':>8} {'rho':>8}")
        for metric in TEXT_METRICS:
            print(
                f"{metric:<10} "
                f"{results['kendall'][metric]:>+8.3f} "
                f"{results['spearman'][metric]:>+8.3f}"
            )

        avg_tau = sum(results["kendall"][m] for m in TEXT_METRICS) / len(TEXT_METRICS)
        avg_rho = sum(results["spearman"][m] for m in TEXT_METRICS) / len(TEXT_METRICS)

        print(f"{'TEXT_AVG':<10} {avg_tau:>+8.3f} {avg_rho:>+8.3f}")

    except Exception as e:
        print(f"\nSkipping local scoring because it failed: {e}")


if __name__ == "__main__":
    main()