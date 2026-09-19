# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair>=5,<7",
#     "fsspec[http]>=2025.3",
#     "jevframe[pandas]==0.1.0",
#     "marimo>=0.24,<1",
#     "typesafe-sdk>=0.7,<0.8",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import os

    import altair as alt
    import marimo as mo
    import pandas as pd
    from typesafe_sdk import Choice, Noul

    import jevframe.pandas  # noqa: F401
    from jevframe import MemoryCache


@app.cell(hide_code=True)
def brand_banner():
    # Official jevframe SVG is embedded so this notebook also works as a standalone file.
    # Cloud artwork from https://typesafe.ai/
    mo.Html("""
    <style>
    .jevframe-brand { position:relative; isolation:isolate; overflow:hidden; border:1px solid
    #202020;
        background:#f8ebf7; color:#181818; font-family:Arial,Helvetica,sans-serif; }
    .jevframe-brand .clouds { position:absolute; inset:0; width:100%; height:100%;
        object-fit:cover; object-position:center top; z-index:-1; }
    .jevframe-brand .bar { display:flex; justify-content:space-between; gap:12px;
    flex-wrap:wrap;
        padding:9px 18px; border-bottom:1px solid #202020; background:#f4f4f0;
        font:11px/1.3 ui-monospace,SFMono-Regular,Consolas,monospace; letter-spacing:.07em; }
    .jevframe-brand .body { padding:24px 26px 26px; }
    .jevframe-brand .brands { display:flex; flex-direction:column;
        align-items:flex-start; gap:4px; }
    .jevframe-brand .logo { display:block; width:340px; max-width:100%; }
    .jevframe-brand .logo svg { display:block; width:100%; height:auto; }
    .jevframe-brand .partner { padding-left:12px; }
    .jevframe-brand .name { color:#181818; text-decoration:none;
        font-size:20px; font-weight:600; line-height:1.3; letter-spacing:-.015em; }
    .jevframe-brand .tagline { margin:16px 0 22px; max-width:38em;
        font-size:17px; line-height:1.5; color:#181818; }
    .jevframe-brand .links { display:flex; gap:24px; flex-wrap:wrap;
        font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; }
    .jevframe-brand .links a { color:#181818; text-decoration:underline;
    text-underline-offset:4px; }
    .jevframe-brand a:focus-visible { outline:2px solid #181818; outline-offset:5px; }
    @media (prefers-reduced-motion:reduce) { .jevframe-brand .clouds { display:none; } }
    </style>

    <section class="jevframe-brand" aria-label="jevframe and TypeSafe AI">
        <img class="clouds"
        src="https://framerusercontent.com/images/P5deyaXUiEjZxxqxpUI5Q9Kgg.gif?height=540&amp;width=1440"
             alt="" aria-hidden="true" width="1440" height="540" referrerpolicy="no-referrer">
        <div class="bar"><span>JEVFRAME × TYPESAFE AI</span><span>SEMANTIC
        DATAFRAMES</span></div>
        <div class="body">
            <div class="brands">
                <a class="logo" href="https://github.com/ktaletsk/jevframe"
                   aria-label="jevframe on GitHub" target="_blank" rel="noopener noreferrer">
                    <svg xmlns="http://www.w3.org/2000/svg" width="693" height="192" viewBox="0
                    0 693 192" role="img" aria-labelledby="title desc">
      <title id="title">jevframe — df.jev()</title>
      <desc id="desc">Two isometric corners, separated horizontally, frame the code
      df.jev().</desc>
      <!-- Original TypeSafe AI path, unchanged; clipped and translated to separate the corners.
      -->
      <defs>
        <path id="typesafe-original-mark" fill-rule="evenodd" d="M76.4576
        17.5651V42.3984L98.8152 56.9225L98.8213 111.911L49.4046 144L22.3637 126.441V100.92L0
        86.3963V32.0953L2.12872 30.7086L49.4106 0L76.4576 17.5651ZM35.6044 123.899L49.3985
        132.858L85.5624 109.362L71.7744 100.41L35.6044 123.899ZM54.0756 61.9949V86.4024L31.7058
        100.927V115.292L67.1155 92.296V53.5286L54.0756 61.9949ZM76.4576 92.3082L89.4732
        100.762V61.9888L76.4576 53.5347V92.3082ZM13.2467 83.8601L27.0347 92.8191L40.8289
        83.8601L27.0408 74.9072L13.2467 83.8601ZM9.34204 37.1617V75.26L22.3637
        66.7998V42.3923L44.7396 27.8561V14.1652L9.34204 37.1617ZM31.7058 66.7937L44.7335
        75.2539V61.9888L31.7058 53.5286V66.7937ZM35.6044 44.9225L49.4106 53.8875L63.2048
        44.9285L49.4046 35.9635L35.6044 44.9225ZM54.0817 27.8561L67.1155 36.3223V22.6376L54.0817
        14.1652V27.8561Z"/>
        <clipPath id="left-piece" clipPathUnits="userSpaceOnUse"><path d="M-1.000 -1.000 L76.458
        -1.000 L76.458 47.463 L54.076 61.995 L54.076 86.402 L27.040 103.957 L0.000 86.396
        L-1.000 86.396 Z"/></clipPath>
        <clipPath id="right-piece" clipPathUnits="userSpaceOnUse"><path d="M71.780 39.359
        L100.000 57.000 L100.000 146.000 L20.000 146.000 L22.364 126.441 L22.364 95.853 L44.734
        81.324 L44.733 56.925 Z"/></clipPath>
      </defs>
      <g fill="#1e1e1e">
        <g id="left-corner" transform="translate(24.000 24.000)">
          <use href="#typesafe-original-mark" clip-path="url(#left-piece)"/>
        </g>
        <g id="right-corner" transform="translate(569.613 24.000)">
          <use href="#typesafe-original-mark" clip-path="url(#right-piece)"/>
        </g>
      </g>
      <path id="wordmark" fill="#1e1e1e" d="M174.579 81.369 L174.579 55.803 L187.126 55.803
      L187.126 122.662 L174.579 122.662 L174.579 115.529 Q172.559 119.697 169.358 121.803
      Q166.157 123.908 161.817 123.908 Q153.567 123.908 149.012 117.291 Q144.458 110.674 144.458
      98.643 Q144.458 86.439 149.077 79.908 Q153.696 73.377 162.247 73.377 Q166.114 73.377
      169.186 75.375 Q172.258 77.373 174.579 81.369 Z M157.047 98.729 Q157.047 105.689 159.368
      109.643 Q161.688 113.596 165.770 113.596 Q169.852 113.596 172.215 109.643 Q174.579 105.689
      174.579 98.729 Q174.579 91.768 172.215 87.814 Q169.852 83.861 165.770 83.861 Q161.688
      83.861 159.368 87.814 Q157.047 91.768 157.047 98.729 Z M225.325 70.326 L225.325 74.537
      L240.106 74.537 L240.106 84.205 L225.325 84.205 L225.325 122.662 L212.735 122.662 L212.735
      84.205 L201.047 84.205 L201.047 74.537 L212.735 74.537 L212.735 71.186 Q212.735 62.506
      216.344 59.154 Q219.954 55.803 229.751 55.803 L240.106 55.803 L240.106 65.471 L230.266
      65.471 Q227.430 65.471 226.420 66.502 Q225.411 67.533 225.325 70.326 Z M265.844 106.893
      L280.153 106.893 L280.153 122.662 L265.844 122.662 Z M336.055 120.814 Q336.055 131.643
      332.167 136.262 Q328.278 140.881 319.254 140.881 L305.676 140.881 L305.676 131.213
      L315.731 131.213 Q319.942 131.213 321.704 128.850 Q323.465 126.486 323.465 120.814
      L323.465 84.205 L310.704 84.205 L310.704 74.537 L336.055 74.537 Z M336.055 65.814 L323.465
      65.814 L323.465 51.119 L336.055 51.119 Z M399.864 120.299 Q395.481 122.104 390.926 123.006
      Q386.372 123.908 381.301 123.908 Q369.227 123.908 362.846 117.441 Q356.465 110.975 356.465
      98.814 Q356.465 87.041 362.610 80.209 Q368.754 73.377 379.368 73.377 Q390.067 73.377
      395.975 79.715 Q401.883 86.053 401.883 97.568 L401.883 102.682 L369.270 102.682 Q369.313
      108.354 372.622 111.146 Q375.930 113.939 382.504 113.939 Q386.844 113.939 391.055 112.693
      Q395.266 111.447 399.864 108.740 Z M389.208 93.229 Q389.122 88.244 386.651 85.688 Q384.180
      83.131 379.368 83.131 Q375.028 83.131 372.450 85.773 Q369.872 88.416 369.399 93.271 Z
      M455.036 74.537 L439.739 122.662 L424.227 122.662 L408.930 74.537 L421.692 74.537 L431.962
      112.092 L442.274 74.537 Z M498.176 55.889 Q492.504 66.158 489.754 75.783 Q487.004 85.408
      487.004 95.033 Q487.004 104.572 489.754 114.262 Q492.504 123.951 498.176 134.264 L488.379
      134.264 Q481.547 124.338 478.239 114.691 Q474.930 105.045 474.930 95.033 Q474.930 85.064
      478.260 75.375 Q481.590 65.686 488.379 55.889 Z M524.731 55.889 L534.528 55.889 Q541.317
      65.686 544.647 75.375 Q547.977 85.064 547.977 95.033 Q547.977 105.045 544.669 114.691
      Q541.360 124.338 534.528 134.264 L524.731 134.264 Q530.403 123.951 533.153 114.262
      Q535.903 104.572 535.903 95.033 Q535.903 85.408 533.153 75.783 Q530.403 66.158 524.731
      55.889 Z"/>
    </svg>
                </a>
                <div class="partner">
                    <a class="name" href="https://typesafe.ai/"
                       target="_blank" rel="noopener noreferrer">Powered by Typesafe AI</a>
                </div>
            </div>
            <p class="tagline"><strong>jevframe</strong> brings TypeSafe AI’s
            <strong>Jev</strong>
                to pandas and Polars.<br>Ask natural-language questions. Get structured
                decisions for every row.</p>
            <nav class="links" aria-label="jevframe and TypeSafe resources">
                <a href="https://github.com/ktaletsk/jevframe" target="_blank" rel="noopener
                noreferrer">jevframe on GitHub ↗</a>
                <a href="https://typesafe.ai/" target="_blank" rel="noopener noreferrer">Meet
                TypeSafe AI ↗</a>
                <a href="https://docs.typesafe.ai/" target="_blank" rel="noopener
                noreferrer">Jev docs ↗</a>
            </nav>
        </div>
    </section>
    """)
    return


@app.cell
def _():
    mo.md("""
    # Semantic questions over real customer reviews

    Turn customer feedback into columns you can filter, sort, and plot with
    `jevframe`. Ask whether a customer is **dissatisfied**, whether they report a
    **product defect**, and **what the review is about**.

    Edit the questions, then click **Evaluate reviews** to see the answers and
    their full probability distributions. Choose separate columns or one
    structured column containing the same answers.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Typesafe AI key

    - **Locally**: `export TYPESAFE_API_KEY='your-key'` before starting marimo
    - **On molab:** open Development Panel in the bottom left corner, click
      the **Secrets** tab and add `TYPESAFE_API_KEY`.
    """)
    return


@app.cell
def review_storage():
    import fsspec

    # Keep the filesystem as a notebook variable for marimo's Remote Storage panel.
    # Cache the small archive locally: UCI's HTTP response is not seekable.
    review_files = fsspec.filesystem(
        "zip",
        fo=(
            "simplecache::https://archive.ics.uci.edu/static/public/331/"
            "sentiment+labelled+sentences.zip"
        ),
        target_options={"https": {"timeout": 30}},
    )
    return (review_files,)


@app.cell
def _(review_files):
    import csv

    @mo.cache
    def load_public_reviews():
        with review_files.open(
            "sentiment labelled sentences/amazon_cells_labelled.txt", "rb"
        ) as source:
            frame = pd.read_csv(
                source,
                sep="\t",
                names=["review", "reference_sentiment"],
                quoting=csv.QUOTE_NONE,
                dtype={"review": "string", "reference_sentiment": "int64"},
            )
        if len(frame) != 1000 or set(frame["reference_sentiment"]) != {0, 1}:
            raise ValueError("The UCI download does not match the expected review dataset.")
        frame["reference_sentiment"] = frame["reference_sentiment"].map(
            {0: "negative", 1: "positive"}
        )
        frame.index = pd.RangeIndex(1, len(frame) + 1, name="review_id")
        return frame

    all_reviews = load_public_reviews().sample(frac=1, random_state=42)
    cache = MemoryCache(max_entries=10_000)

    mo.md("""
    ## Real customer feedback

    **1,000 Amazon review excerpts** from
    [UCI's Sentiment Labelled Sentences](https://doi.org/10.24432/C57604).
    Start with 50 rows, or increase the sample below. The shuffle is fixed, so
    larger samples include the same earlier rows and can reuse cached answers.

    Browse the source files under **Files → Remote Storage → review_files**.
    The public ZIP is downloaded once to a temporary cache by fsspec.

    The original sentiment labels are shown for comparison; **only the review
    text is sent to Jev**. This dataset deliberately contains 500 positive and
    500 negative excerpts, so it does not estimate the rate of unhappy customers.

    Source: Kotzias et al. (2015), *From Group to Individual Labels using Deep
    Features*. Dataset licensed under
    [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
    """)
    return all_reviews, cache


@app.cell(hide_code=True)
def sample_size_control(all_reviews):
    sample_size = mo.ui.slider(
        start=10,
        stop=len(all_reviews),
        step=10,
        value=50,
        label="Reviews to evaluate",
        show_value=True,
        full_width=True,
    )
    sample_size  # noqa: B018
    return (sample_size,)


@app.cell(hide_code=True)
def sample_reviews(all_reviews, sample_size):
    # Ordinary pandas row selection; preserve each review's original row number.
    reviews = all_reviews.head(sample_size.value).copy()
    mo.vstack(
        [
            mo.md(
                f"**{len(reviews):,} reviews selected.** Click **Evaluate reviews** below "
                "to ask all three questions. Each uncached review uses one request."
            ),
            mo.ui.table(reviews, selection=None, page_size=5),
        ]
    )
    return (reviews,)


@app.cell
def _():
    question = mo.ui.text_area(
        value="Is this customer dissatisfied?",
        label="Dissatisfaction · yes/no probability",
        full_width=True,
    )
    defect_question = mo.ui.text_area(
        value=(
            "Does this review report that the product is broken, unreliable, "
            "or does not work as expected?"
        ),
        label="Product defect · yes/no probability",
        full_width=True,
    )
    topic_question = mo.ui.text_area(
        value="What is the main topic of this review?",
        label="Topic · functionality / usability / value / service / other",
        full_width=True,
    )
    evaluate = mo.ui.run_button(label="Evaluate reviews")
    output_layout = mo.ui.dropdown(
        options={"Separate columns": "columns", "One structured column": "struct"},
        value="Separate columns",
        label="Output layout",
    )
    mo.vstack([question, defect_question, topic_question, output_layout, evaluate])
    return defect_question, evaluate, output_layout, question, topic_question


@app.cell
async def _(
    cache,
    defect_question,
    evaluate,
    output_layout,
    question,
    reviews,
    topic_question,
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
        not all(_input.value.strip() for _input in [question, defect_question, topic_question]),
        mo.md("Enter all three questions first."),
    )
    with mo.status.progress_bar(total=len(reviews), title="Evaluating reviews") as _bar:
        decisions = await reviews.jev.evaluate(
            # Keep the reference labels out of the model's context.
            state="review",
            questions={
                "dissatisfied": Noul(instructions=question.value),
                "defect": Noul(instructions=defect_question.value),
                "topic": Choice(
                    instructions=topic_question.value,
                    criteria={
                        "functionality": (
                            "Performance, reliability, defects, or whether the product works"
                        ),
                        "usability": "Ease of use, setup, comfort, fit, or design",
                        "value": "Price, affordability, or value for money",
                        "service": "Customer support, delivery, seller service, or returns",
                        "other": "General praise, recommendations, or another topic",
                    },
                ),
            },
            output=output_layout.value,
            cache=cache,
            errors="coerce",
            max_concurrency=4,
            progress=lambda update: _bar.update(),
        )
    results = pd.concat([reviews, decisions], axis=1)
    _failed = decisions.isna().all(axis=1)
    results.insert(0, "evaluation_status", _failed.map({False: "ok", True: "failed"}))
    _message = (
        f"**{len(results) - int(_failed.sum()):,} of {len(results):,} "
        "reviews evaluated successfully.**"
    )
    if _failed.any():
        _message += (
            f" {int(_failed.sum()):,} failed rows are marked in the table and keep missing "
            "probabilities. The histogram includes valid results only. "
            "See the evaluation warning for failure positions."
        )
    mo.md(_message).callout(kind="warn" if _failed.any() else "success")
    return decisions, results


@app.cell
def _(results):
    mo.ui.table(results, selection=None)
    return


@app.cell
def _(decisions):
    # Unpack for plotting only; inference and the result table use the selected layout.
    _columns = (
        pd.json_normalize(decisions["result"].tolist()).set_axis(decisions.index)
        if "result" in decisions.columns
        else decisions
    )
    mo.stop(
        _columns.empty or _columns.dropna(how="all").empty,
        mo.md("No valid results to plot. Check the evaluation warning before trying again."),
    )
    _distribution = _columns.melt(
        value_vars=["dissatisfied__probability", "defect__probability"],
        var_name="question",
        value_name="probability",
    )
    _distribution["question"] = _distribution["question"].map(
        {"dissatisfied__probability": "Dissatisfied", "defect__probability": "Product defect"}
    )
    alt.Chart(_distribution.dropna(subset=["probability"])).mark_bar(color="#267f79").encode(
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
def _(reviews):
    mo.md(f"""
    ### One row, one request, several answers

    Each review is sent in one request containing all three questions. This demo
    runs up to **4 row requests concurrently**: **{len(reviews):,} selected reviews**
    need at most **{len(reviews):,} initial requests**, not {3 * len(reviews):,}.
    Retries can add requests; cached reviews need none.

    Each result keeps its input index and includes `dissatisfied__probability`,
    `defect__probability`, and the selected topic with its full distribution.
    The original `reference_sentiment` column remains available for comparison.

    `output="columns"` returns separate fields. `output="struct"` packs those
    same fields into one `result` column. Both keep one output row per review.
    This demo uses `errors="coerce"`: a failed review stays in the table with
    missing answers, an explicit status, and a warning. The library's default
    `errors="raise"` stops on the first failure.

    Changing only the layout or increasing the sample size reuses cached answers
    for reviews already evaluated with the same questions.
    """)
    return


if __name__ == "__main__":
    app.run()
