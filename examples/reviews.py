# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair>=5,<7",
#     "jevframe[pandas]==0.1.0",
#     "marimo>=0.24,<1",
#     "typesafe-sdk>=0.7,<0.8",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import altair as alt
    import marimo as mo
    import pandas as pd
    from typesafe_sdk import Choice, Noul

    import jevframe.pandas  # noqa: F401
    from jevframe import MemoryCache

    return Choice, MemoryCache, Noul, alt, mo, os, pd


@app.cell
def _(mo):
    mo.md("""
    # Semantic questions over a dataframe

    Ask three questions about each review: **dissatisfied?**, **urgent?**, and
    **what topic?** Edit the questions, then click **Evaluate reviews** to see
    their results and probability distributions. Choose separate columns or one
    structured column containing the same answers.

    **On molab:** fork the notebook into your workspace, open **Secrets** in the
    sidebar, and add `TYPESAFE_API_KEY`. Locally, export the variable before
    starting marimo. The SDK reads the key from the environment.

    Evaluations run only when you click the button. No key belongs in notebook code.
    """)
    return


@app.cell
def _(MemoryCache, pd):
    reviews = pd.DataFrame(
        {
            "title": [
                "Double charge",
                "Love it",
                "Cannot log in",
                "Nice update",
                "Still waiting",
                "A small request",
                "Broken export",
                "All good",
            ],
            "review": [
                "I was charged twice. Please refund the extra charge today.",
                "Everything works beautifully. Thanks for the quick setup!",
                "My whole team is locked out and we have a deadline this afternoon.",
                "The new dashboard is much easier to use.",
                "I contacted support last week and nobody has replied.",
                "Could you add a dark theme? The product works well otherwise.",
                "The export crashes every time. I need these reports for tomorrow.",
                "The issue was fixed quickly and I am happy with the resolution.",
            ],
        }
    )
    cache = MemoryCache(max_entries=100)
    reviews  # noqa: B018
    return cache, reviews


@app.cell
def _(mo):
    question = mo.ui.text_area(
        value="Is this customer dissatisfied?",
        label="Dissatisfaction · yes/no probability",
        full_width=True,
    )
    urgency_question = mo.ui.text_area(
        value="Does this customer need urgent help?",
        label="Urgency · yes/no probability",
        full_width=True,
    )
    topic_question = mo.ui.text_area(
        value="What is the main issue in this review?",
        label="Topic · billing / bug / other",
        full_width=True,
    )
    evaluate = mo.ui.run_button(label="Evaluate reviews")
    output_layout = mo.ui.dropdown(
        options={"Separate columns": "columns", "One structured column": "struct"},
        value="Separate columns",
        label="Output layout",
    )
    mo.vstack([question, urgency_question, topic_question, output_layout, evaluate])
    return evaluate, output_layout, question, topic_question, urgency_question


@app.cell
async def _(
    Choice,
    Noul,
    cache,
    evaluate,
    mo,
    os,
    output_layout,
    pd,
    question,
    reviews,
    topic_question,
    urgency_question,
):
    mo.stop(not evaluate.value)
    mo.stop(
        not os.environ.get("TYPESAFE_API_KEY", "").strip(),
        mo.md(
            "Add **TYPESAFE_API_KEY** in molab's **Secrets** sidebar, then click "
            "**Evaluate reviews** again. Locally, export the variable before starting marimo."
        ).callout(kind="warn"),
    )
    mo.stop(
        not all(_input.value.strip() for _input in [question, urgency_question, topic_question]),
        mo.md("Enter all three questions first."),
    )
    with mo.status.progress_bar(total=len(reviews), title="Evaluating reviews") as _bar:
        decisions = await reviews.jev.evaluate(
            state=["title", "review"],
            questions={
                "dissatisfied": Noul(instructions=question.value),
                "urgent": Noul(instructions=urgency_question.value),
                "topic": Choice(
                    instructions=topic_question.value,
                    criteria={
                        "billing": "Charges, payments, or refunds",
                        "bug": "Broken features or access problems",
                        "other": "Praise, requests, or anything else",
                    },
                ),
            },
            output=output_layout.value,
            cache=cache,
            max_concurrency=4,
            progress=lambda update: _bar.update(),
        )
    results = pd.concat([reviews, decisions], axis=1)
    return decisions, results


@app.cell
def _(mo, results):
    mo.ui.table(results, selection=None)
    return


@app.cell
def _(alt, decisions, pd):
    # Unpack for plotting only; inference and the result table use the selected layout.
    _columns = (
        pd.DataFrame(decisions["result"].tolist(), index=decisions.index)
        if "result" in decisions.columns
        else decisions
    )
    _distribution = _columns.melt(
        value_vars=["dissatisfied__probability", "urgent__probability"],
        var_name="question",
        value_name="probability",
    )
    _distribution["question"] = _distribution["question"].map(
        {"dissatisfied__probability": "Dissatisfied", "urgent__probability": "Urgent"}
    )
    alt.Chart(_distribution).mark_bar(color="#267f79").encode(
        x=alt.X(
            "probability:Q",
            bin=alt.Bin(step=0.1, extent=[0, 1]),
            scale=alt.Scale(domain=[0, 1]),
            title="Probability of yes",
        ),
        y=alt.Y("count():Q", title="Reviews", axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip("count():Q", title="Reviews")],
    ).properties(width=260, height=240).facet(column=alt.Column("question:N", title=None))
    return


@app.cell
def _(mo):
    mo.md("""
    ### One row, one request, several answers

    ```text
    review 1 ── one request with all 3 questions ── result row 1
    review 2 ── one request with all 3 questions ── result row 2
       …                                            …
    review 8 ── one request with all 3 questions ── result row 8
    ```

    This demo runs up to **4 row requests concurrently**. Each output row keeps
    its input position and gains `dissatisfied__probability`, `urgent__probability`,
    and topic columns: `topic__label`, `topic__confidence`, and one probability per
    topic. Eight rows need eight initial requests, not 24. Retries can add requests;
    cached rows need none. Repeating the same questions reuses the memory cache.

    `output="columns"` returns these fields as separate columns. `output="struct"`
    packs the same fields into one `result` column. Both layouts keep one output
    row per review and include every probability. Switching layouts and evaluating
    again reuses the same cached responses.
    """)
    return


if __name__ == "__main__":
    app.run()
