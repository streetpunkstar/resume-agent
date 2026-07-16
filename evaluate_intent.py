"""
Runs the 'intent-classification' Langfuse dataset against the current
classify_intent() node and records the results as a Dataset Run — viewable
and comparable in the Langfuse UI under the dataset's Runs tab.

Run from the project root, in your MAIN environment (not crewai-env):
    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_SECRET_KEY="sk-lf-..."
    export LANGFUSE_HOST="http://localhost:3000"
    python3 eval_classify_intent.py

Install (if not already present):
    pip install langfuse
"""
from datetime import datetime
from langfuse import get_client, Evaluation
from graph.pipeline import classify_intent


def extract_message(item_input) -> str:
    """Dataset items added via 'Add to dataset' from a trace observation can
    land in a couple of different shapes depending on what was captured.
    Handles the common cases; prints the raw input if none match, so you can
    see exactly what's there and adjust this function accordingly."""
    if isinstance(item_input, str):
        return item_input
    if isinstance(item_input, dict):
        if "message" in item_input:
            return item_input["message"]
        for key in ("state", "input"):
            nested = item_input.get(key)
            if isinstance(nested, dict) and "message" in nested:
                return nested["message"]
    raise ValueError(
        f"Couldn't find a message in this dataset item's input: {item_input!r}\n"
        f"Check the dataset item in the Langfuse UI and adjust extract_message() to match."
    )


def task(*, item, **kwargs):
    """Runs classify_intent on one dataset item's message, returns the
    predicted intent — this is the thing being tested."""
    message = extract_message(item.input)
    result = classify_intent({"message": message})
    return result["intent"]


def extract_expected_intent(expected_output) -> str:
    """expected_output was auto-captured from the full trace state, not just
    the intent label — it's a dict with an 'intent' key, not a bare string."""
    if isinstance(expected_output, str):
        return expected_output
    if isinstance(expected_output, dict) and "intent" in expected_output:
        return expected_output["intent"]
    raise ValueError(f"Couldn't find an expected intent in: {expected_output!r}")


def intent_match(*, input, output, expected_output, **kwargs):
    """Scores one item: did the predicted intent match what you annotated
    as correct? Returns an Evaluation, which becomes a Score in Langfuse."""
    expected_intent = extract_expected_intent(expected_output)
    is_correct = output == expected_intent
    return Evaluation(
        name="intent_correct",
        value=1.0 if is_correct else 0.0,
        comment=f"predicted={output!r}, expected={expected_intent!r}",
    )


if __name__ == "__main__":
    langfuse = get_client()
    dataset = langfuse.get_dataset("intent_classification")

    run_name = f"classify_intent-check-{datetime.now().strftime('%Y%m%d-%H%M')}"
    result = dataset.run_experiment(
        name=run_name,
        description="Regression check for classify_intent after the interview-keyword fix",
        task=task,
        evaluators=[intent_match],
    )

    print(result.format(include_item_results=True))

    print("\n--- Raw comparison, item by item ---")
    for item_result in result.item_results:
        print(f"input:    {item_result.item.input!r}")
        print(f"output:   {item_result.output!r}")
        print(f"expected: {item_result.item.expected_output!r}")
        print("-" * 40)
