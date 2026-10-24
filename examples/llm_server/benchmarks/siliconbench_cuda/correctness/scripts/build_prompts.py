#!/usr/bin/env python3
"""
Build prompts for the GMRID_v3 classification task, mirroring
inflaton/llms-at-edge prompt rendering byte-for-byte so reference F1 is
comparable to their paper.

Reads:
  data/GMRID_v3-test.csv  (Headline_Details = input, Summarized_label = gold)
  data/categories.json    (8-class taxonomy with descriptions)

Writes:
  prompts/<N>shot/prompts.jsonl  — {id, messages, max_tokens}
  prompts/<N>shot/labels.jsonl   — {id, gold_category}

Qwen3 thinking-off portability: appends `/no_think` to the user turn. Works
across any backend that respects the Qwen3 chat template; doesn't require
framework-specific `chat_template_kwargs` plumbing.
"""

import argparse
import csv
import json
from pathlib import Path

# System prompt template — copied verbatim from inflaton/llms-at-edge
# llm_toolkit/data_utils.py (system_prompt_template), then rendered the same way.
SYSTEM_PROMPT_TEMPLATE = """Task: Classify Inputs into Predefined Categories

Your primary objective is to analyze the given input and assign it to one of the predefined categories: {categories_list}. Evaluate the content carefully and use the defining characteristics of each category to ensure an accurate classification.

Guidelines:
1. Understand the Categories:
Each category has specific attributes that distinguish it. Familiarize yourself with these attributes by referring to the category descriptions provided in the JSON below. Use these details to guide your classification:

{categories_json}

2. Contextual Analysis:
Consider the broader context of the input. If an input could potentially fit into multiple categories, select the one that most closely aligns with its primary intent or focus.
3. Handling Ambiguity:
For ambiguous inputs or those that do not clearly align with any category, choose the category that most closely matches the content provided.
4. Ensure Accuracy and Consistency:
Strive for consistent and accurate classifications. Avoid arbitrary or random assignments.
5. Provide Feedback:
If the input cannot be classified into any of the given categories, classify it as “Others.”

Instructions for Output:
1. Once the category is identified, provide “specific tags” by selecting from the list corresponding to the identified category, as defined in the JSON.
2. Ensure the selected “specific tags” accurately reflect the details and context of the input.

Output Format:

Return your classification in the following JSON format:

{{{{
  "category": "<Selected Category>",
  "specific_tags": ["<Selected Tag 1>", "<Selected Tag 2>", ...]
}}}}


"""

# Few-shot examples — copied verbatim from inflaton/llms-at-edge
# llm_toolkit/data_utils.py (gpt_4o_generated_examples).
GPT_4O_GENERATED_EXAMPLES = """
- Input:

Local sources reported that operations at Pier 1 and 2 container terminals at the Port of Durban have suspended due to strong winds on December 27 from 18:50 (local time) and resumed at 23:10 on the same day. For Pier 2 terminal, operations stopped at 19:30 and resumed at 20:35 respectively.

- Output:

{{
  "category": "Weather",
  "specific_tags": ["Severe Winds"]
}}

- Input:

Information received states that emergency personnel are working to contain a blaze at Off Road Warehouse in commercial San Diego, on 17 November. It is detailed that the store is located at 7915 Balboa Avenue. Traffic maps show that Balboa Avenue is closed both ways between Mercury Street and Convoy Street. Travelers should use caution in the area and divert away from any encountered fire suppression operations.

- Output:

{{
  "category": "Administrative Issue",
  "specific_tags": ["Roadway Closure", "Public Safety Advisory"]
}}

- Input:

Protests against climate change are anticipated nationwide on 29 November and 6 December as part of the ‘Fridays for Future’ global climate strike. Specific details of planned events have not been confirmed, but are likely to occur in major cities across the country. Previous climate strikes have seen large turnout in cities such as New York City, Philadelphia, and Washington, D.C.

- Output:

{{
  "category": "Worker Strike",
  "specific_tags": ["Protest", "Civil Unrest Advisory"]
}}

- Input:

Government sources reported a fire at the Woolwich Dockyard, located near Belson Rd and Borgard Rd. No injuries were immediately reported. All rail lines from London towards Slade Green are running again. This incident is closed.

- Output:

{{
  "category": "Accident",
  "specific_tags": ["Non-industrial Fire"]
}}

- Input:

Local media sources indicated on November 30 that the Ekurhuleni Central Crime Intelligence Unit arrested 4 suspects and recovered computer printer equipment cargo from their November 21 truck theft at the corner of Main Reef Road and Ulysses Street in Cleveland. The truck was en route from Durban to Johannesburg when it was hijacked in Randfontein. The cargo was worth ZAR 5 million (EUR 309018.21; USD 352673.95), and some laptops are still missing. Distributors should be mindful of cargo theft risks in Randfontein and should plan accordingly.

- Output:

{{
  "category": "Terrorism",
  "specific_tags": ["Cargo Theft", "Organized Crime"]
}}

- Input:

Anonymous sources have reported that a ransomware attack has disrupted network operations for a major logistics provider. The attack occurred on November 15, and data breaches were confirmed, exposing sensitive customer and shipment details. The company has stated that recovery is underway but advised customers to expect delays.

- Output:

{{
  "category": "Cyber Attack",
  "specific_tags": ["Ransomware", "Data Breach"]
}}

- Input:

The Selangor Health Department reported that two students of a Secondary School in Pandamaran Jaya in Port Klang had been infected with COVID-19 virus.

- Output:

{{
  "category": "Others",
  "specific_tags": ["Outbreak of Disease"]
}}

- Input:

An incident of workplace negligence was reported at a construction site in downtown Chicago on November 19, where an unfastened scaffolding collapsed, injuring two workers. Investigations are ongoing to determine accountability.

- Output:

{{
  "category": "Human Error",
  "specific_tags": ["Workplace Accident"]
}}

- Input:

Shipping delays were reported at the Port of Los Angeles on December 1 due to a customs system outage. Containers requiring clearance were delayed for up to 12 hours, affecting supply chains across the region.

- Output:

{{
  "category": "Administrative Issue",
  "specific_tags": ["Customs Delay", "Port Disruption"]
}}

- Input:

Russian media sources are reporting that courts, schools, and hospitals across Saint Petersburg have been evacuated today due to anonymous threats. It is understood that people have been evacuated from Petrodvorets, Oktyabrsky, Kolpinsky, Petrogradsky, Kuibyshevsky, and Sestroretsky district courts. Furthermore, the State University of the Sea and River Fleet, St. Petersburg State University of Railway Engineering, Higher School of Folk Arts, St. Petersburg State University of Telecommunications, and S.M. Military Medical Academy Kirov have all been evacuated. This is the fourth consecutive week of evacuations from public buildings due to such threats. It is not known when the situation will normalize.

- Output:

{{
  "category": "Terrorism",
  "specific_tags": ["Bomb Threat", "Public Safety"]
}}

- Input:

A series of earthquakes shook the southern region of California on November 22. The most powerful tremor measured 6.4 on the Richter scale, causing landslides in rural areas and minor structural damage in towns nearby. Emergency services are on alert.

- Output:

{{
  "category": "Weather",
  "specific_tags": ["Earthquake", "Landslide"]
}}

"""


def find_nth_occurrence(string, substring, n):
    """Replicates inflaton's find_nth_occurrence."""
    start = -1
    for _ in range(n):
        start = string.find(substring, start + 1)
        if start == -1:
            return -1
    return start


def render_system_prompt(categories_json: dict, num_shots: int) -> str:
    """
    Render the system prompt the same way inflaton's get_prompt_templates does
    with remove_double_curly_brackets=True.
    """
    categories_list = list(categories_json.keys())

    # inflaton builds the substitution value as:
    #   "{" + f"{categories_json}" + "}"
    # which gives {{<dict repr>}} inside the formatted string (the extra braces
    # show up because the template has a single {categories_json} slot; when
    # later replace("{{","{") runs it collapses to a single pair).
    categories_repr = "{" + f"{categories_json}" + "}"

    system = SYSTEM_PROMPT_TEMPLATE.format(
        categories_list=categories_list,
        categories_json=categories_repr,
    )

    if num_shots > 0:
        # Slice to the (N+1)th "- Input:" marker — inflaton's approach.
        index = find_nth_occurrence(
            GPT_4O_GENERATED_EXAMPLES, "- Input:", num_shots + 1
        )
        slice_end = index if index >= 0 else len(GPT_4O_GENERATED_EXAMPLES)
        system += "\nExample Inputs and Outputs:\n" + GPT_4O_GENERATED_EXAMPLES[:slice_end]

    # remove_double_curly_brackets=True — collapse {{ → { and }} → }
    system = system.replace("{{", "{").replace("}}", "}")
    return system


def build_user_prompt(headline_details: str, no_think: bool) -> str:
    user = f"- Input:\n\n{headline_details}\n\n- Output:\n\n"
    if no_think:
        user += "/no_think"
    return user


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, required=True,
                        help="Number of few-shot examples (0 or e.g. 5).")
    parser.add_argument("--csv", default=None,
                        help="Path to GMRID_v3-test.csv (default: ../data/).")
    parser.add_argument("--categories", default=None,
                        help="Path to categories.json (default: ../data/).")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for prompts.jsonl + labels.jsonl "
                             "(default: ../prompts/<N>shot/).")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Cap number of rows (0 = all). Useful for smoke tests.")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="max_tokens to bake into each prompt record.")
    parser.add_argument("--no-think", dest="no_think", action="store_true", default=True,
                        help="Append /no_think to user turn (default: on).")
    parser.add_argument("--think", dest="no_think", action="store_false",
                        help="Leave thinking mode enabled (for A/B comparison).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    csv_path = Path(args.csv) if args.csv else data_dir / "GMRID_v3-test.csv"
    cats_path = Path(args.categories) if args.categories else data_dir / "categories.json"

    out_dir = (Path(args.output_dir) if args.output_dir
               else script_dir.parent / "prompts" / f"{args.shots}shot")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cats_path) as f:
        categories_json = json.load(f)
    category_set = set(categories_json.keys())

    system_prompt = render_system_prompt(categories_json, args.shots)

    prompts_path = out_dir / "prompts.jsonl"
    labels_path = out_dir / "labels.jsonl"

    n_written = 0
    n_skipped_no_input = 0
    n_skipped_bad_label = 0

    with open(csv_path, newline="") as f_csv, \
            open(prompts_path, "w") as f_prompts, \
            open(labels_path, "w") as f_labels:
        reader = csv.DictReader(f_csv)
        for row in reader:
            headline = row["Headline_Details"].strip()
            gold = row["Summarized_label"].strip()
            row_id = row["id"]

            if not headline:
                n_skipped_no_input += 1
                continue
            if gold not in category_set:
                n_skipped_bad_label += 1
                continue

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(headline, args.no_think)},
            ]
            f_prompts.write(json.dumps({
                "id": row_id,
                "messages": messages,
                "max_tokens": args.max_tokens,
            }, ensure_ascii=False) + "\n")
            f_labels.write(json.dumps({
                "id": row_id,
                "gold_category": gold,
            }, ensure_ascii=False) + "\n")
            n_written += 1

            if args.max_rows and n_written >= args.max_rows:
                break

    print(f"Wrote {n_written} prompts and labels.")
    print(f"  prompts: {prompts_path}")
    print(f"  labels:  {labels_path}")
    if n_skipped_no_input:
        print(f"  skipped {n_skipped_no_input} rows with empty Headline_Details")
    if n_skipped_bad_label:
        print(f"  skipped {n_skipped_bad_label} rows with Summarized_label "
              f"outside canonical taxonomy")
    print(f"System prompt length: {len(system_prompt)} chars "
          f"(~{len(system_prompt) // 4} tokens)")
    print(f"Thinking mode: {'OFF (/no_think appended)' if args.no_think else 'ON'}")


if __name__ == "__main__":
    main()
