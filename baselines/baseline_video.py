import json
import os
import re
import time

import pandas as pd
from datasets import load_from_disk
from dotenv import load_dotenv
from google import genai
from google.genai import types

from utils import score_submission

load_dotenv()

SPLIT = "validation"
SPLITS_PATH = "challenge-dataset"
OUT_PATH = f"output/submission_vlm_video_{SPLIT}.csv"

MODEL = "gemini-3.1-pro-preview"

TEXT_METRICS = ["TCC", "MS", "M"]
VIDEO_METRICS = ["VC", "VA", "VE"]

# Placeholder text scores.
# Text metrics are 0-indexed: 0, 1, 2.
DEFAULT_TEXT_SCORE = 0

RUBRIC = """You are an expert evaluator of educational animations for Computer Science analogies.
Given the target concept, its description, the textual analogy, and an animation video,
score the animation on three metrics. Use these exact 0-indexed scales:

VC - Visual Clarity/Design: how clearly the visual elements represent the scene in the analogy.
  0 = poor, 1 = fair, 2 = good, 3 = excellent

VA - Alignment with Textual Analogy: how completely and accurately the animation covers the analogy text.
  0 = poor, 1 = fair, 2 = good, 3 = excellent

VE - Visual Engagement: how effectively the animation holds attention and fosters interest.
  0 = lacks engagement, 1 = some moments of interest, 2 = generally engaging, 3 = consistently captivates

Respond with ONLY a JSON object, no prose, no code fences:
{"VC": <int>, "VA": <int>, "VE": <int>}
"""


def build_prompt(target, description, analogy):
    return f"""{RUBRIC}

TARGET CONCEPT: {target}

DESCRIPTION: {description}

ANALOGY TEXT:
{analogy}

Watch the animation and score it. JSON only:"""


def parse_scores(text):
    text = re.sub(r"```(?:json)?|```", "", text).strip()

    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in: {text!r}")

    obj = json.loads(match.group(0))

    scores = {}
    for metric in VIDEO_METRICS:
        if metric not in obj:
            raise ValueError(f"Missing metric {metric} in model output: {obj}")

        value = int(obj[metric])

        if value not in [0, 1, 2, 3]:
            raise ValueError(
                f"Invalid value for {metric}: {value}. "
                "Allowed video values are 0, 1, 2, 3."
            )

        scores[metric] = value

    return scores


def resolve_video_path(row):
    """
    Handles common video formats:
    - row["video"] is a relative path string, e.g. test_videos/test_0.mp4
    - row["video"] is a dict with {"path": "..."}
    - row["video"] is already an absolute path
    """
    video = row["video"]

    if isinstance(video, dict):
        video_path = video.get("path")
    else:
        video_path = video

    if video_path is None:
        raise ValueError(f"Could not resolve video path from row['video']: {video!r}")

    if os.path.isabs(video_path):
        return video_path

    return os.path.join(SPLITS_PATH, video_path)


def score_one(client, video_path, target, description, analogy, max_retries=3):
    prompt = build_prompt(target, description, analogy)

    uploaded = client.files.upload(file=video_path)

    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"Video upload failed: {uploaded.state}")

    try:
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(
                    model=MODEL,
                    contents=[uploaded, prompt],
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

        # Fallback: middle-ish of the video scale.
        return {m: 1 for m in VIDEO_METRICS}

    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass


def make_submission_row(example_id, video_scores):
    return {
        "id": example_id,
        "TCC": DEFAULT_TEXT_SCORE,
        "MS": DEFAULT_TEXT_SCORE,
        "M": DEFAULT_TEXT_SCORE,
        "VC": video_scores["VC"],
        "VA": video_scores["VA"],
        "VE": video_scores["VE"],
    }


def main():
    os.makedirs("output", exist_ok=True)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    test = load_from_disk(SPLITS_PATH)[SPLIT]

    rows = []

    for i in range(len(test)):
        row = test[i]
        video_path = resolve_video_path(row)

        print(f"[{i + 1}/{len(test)}] {row['target']} ({video_path})")

        video_scores = score_one(
            client,
            video_path,
            row["target"],
            row["description"],
            row["analogy"],
        )

        submission_row = make_submission_row(i, video_scores)

        print(f"    -> {submission_row}")

        rows.append(submission_row)

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
        for metric in VIDEO_METRICS:
            print(
                f"{metric:<10} "
                f"{results['kendall'][metric]:>+8.3f} "
                f"{results['spearman'][metric]:>+8.3f}"
            )

        avg_tau = sum(results["kendall"][m] for m in VIDEO_METRICS) / len(VIDEO_METRICS)
        avg_rho = sum(results["spearman"][m] for m in VIDEO_METRICS) / len(VIDEO_METRICS)

        print(f"{'VIDEO_AVG':<10} {avg_tau:>+8.3f} {avg_rho:>+8.3f}")

    except Exception as e:
        print(f"\nSkipping local scoring because it failed: {e}")


if __name__ == "__main__":
    main()