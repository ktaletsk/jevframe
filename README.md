# jevframe

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ktaletsk/jevframe/blob/main/examples/reviews.py)

Ask the same semantic questions about every row of a dataframe using
[TypeSafe Jev](https://docs.typesafe.ai/). Get ordinary pandas or Polars results,
with bounded async inference and complete probability distributions.

## Install

```sh
uv add 'jevframe[pandas]==0.1.0'     # in your uv project
# Or: uv add 'jevframe[polars]==0.1.0' / 'jevframe[pandas,polars]==0.1.0'
export TYPESAFE_API_KEY='your-key'
```

Python 3.10+; the official `typesafe-sdk>=0.7,<0.8` handles inference. Dataframe
dependencies are optional. Importing an integration registers its `.jev` accessor.

On **molab**, open the notebook's **Secrets** sidebar panel and add
`TYPESAFE_API_KEY` with your key as its value. molab loads it automatically and
persists it between sessions; `.env` secrets are excluded from forks.
[Molab's secrets documentation](https://marimo.io/pages/molab/storage#secrets-stay-with-your-notebook).
Then click **Evaluate reviews** in the demo. You do not need to put the key in a code cell.

## pandas

These examples use notebook top-level `await`. In a script, put them inside an
`async def main()` and call `asyncio.run(main())`.

```python
import pandas as pd
import jevframe.pandas
from typesafe_sdk import Choice, Noul

df = pd.DataFrame(
    {
        "title": ["Charged twice", "Great update"],
        "review": ["Please refund the duplicate charge today.", "Everything works well!"],
    }
)

df["dissatisfied"] = await df.jev.noul(
    "Is this customer dissatisfied?",
    state=["title", "review"],
)

topics = await df.jev.choice(
    "What is the main issue?",
    choices=["billing", "bug", "other"],
    state="review",
)

grades = await df.jev.score(
    "How positive is this review?",
    levels=["Negative", "Neutral", "Positive"],
    state="review",
)

results = await df.jev.evaluate(
    state=["title", "review"],
    questions={
        "urgent": Noul(instructions="Does the customer need help today?"),
        "topic": Choice(
            instructions="What is the main issue?",
            criteria={"billing": "Charges and refunds", "bug": "Broken behavior", "other": None},
        ),
    },
)
```

| Method | Result columns |
| --- | --- |
| `noul` | Series named `probability`, the probability of yes |
| `choice` | `label`, `confidence`, `p__billing`, `p__bug`, … |
| `score` | `level`, `label`, `score`, `confidence`, `p__0`, `p__1`, … |
| `evaluate` | Fields prefixed by question name, e.g. `urgent__probability`, `topic__p__billing` |

`score` is the expected **zero-based** level, not a probability: with five levels it
ranges from 0 to 4. `level`/`label` identify the highest-probability level, with ties
resolved by level order. Structured SDK level descriptions appear as JSON text in
`label`. Confidence comes from the SDK; it is separate from the selected option's
probability. No result is thresholded or silently renormalized.

Question and criterion order determine column order. Row order and pandas indexes,
including duplicates and MultiIndexes, are preserved. The input is never mutated.

### Output layout

`choice`, `score`, and `evaluate` accept `output="columns"` (the default) or
`output="struct"`. Both return a DataFrame with one row per input row.
The struct layout has a single column named `result`: dictionaries in pandas
(`object` dtype), or a native `Struct` in Polars. Its fields have exactly the same
names, order, and values as the separate columns, including all probabilities.
`evaluate` keeps question prefixes inside the struct too.

```python
packed = await df.jev.evaluate(
    state="review",
    questions={
        "dissatisfied": Noul(instructions="Is this customer dissatisfied?"),
        "urgent": Noul(instructions="Does the customer need help today?"),
    },
    output="struct",
)
df["decisions"] = packed["result"]
# Each successful cell: {"dissatisfied__probability": 0.9, "urgent__probability": 0.8}
# Illustrative probabilities; actual values come from Jev.
```

Failed or skipped rows have a missing `result` cell (`None` in pandas, null in
Polars). Empty results retain their output dtype, including the full Polars struct
schema. Layout affects only presentation: switching it makes no extra requests
when you reuse a cache. `noul` always returns its single probability Series.

## How requests and results map to rows

The engine sends **one input row per request**, with up to 16 row requests in flight
by default (`max_concurrency`). With `evaluate()`, every question about that row
shares the same context and request. Different rows are evaluated independently.

For example, 100 rows with three questions produce 100 initial requests and 100
result rows, with answers in separate columns or one structured column. Retries can add
requests; skipped or cached rows need none. This does not expand one input row
into multiple generated records. The demo evaluates dissatisfaction, urgency,
and topic together for each review.

## Row selection and context

Select rows with the dataframe library before evaluation:

```python
subset = await df.iloc[:10].jev.noul("Is this urgent?", state="review")

custom = await df.jev.noul(
    "Is this urgent?",
    state=lambda row: {"text": f"{row['title']}: {row['review']}"},
)
```

`state="review"` sends `{"review": value}`; a list sends those named columns.
Indexes are never sent implicitly. Selected column names must be unique strings;
unrelated duplicate columns are allowed. A synchronous context callable receives
a detached mapping of all columns and returns text, an object, or an array. It
runs once per row before cache lookup. Polars temporal values in these mappings
are ISO strings to retain nanosecond precision.

Missing scalars become JSON null; lists and objects are normalized recursively.
Dates/times become ISO text. Unsupported objects and infinities raise a row-aware
context error. Supply a callable to convert custom values. Context errors always
raise, even under `errors="coerce"`.

## Eager Polars

```python
import polars as pl
import jevframe.polars

df = pl.DataFrame({"review": ["Charged twice", "Works perfectly"]})
probabilities = await df.jev.noul("Is this customer dissatisfied?", state="review")
df = df.with_columns(probabilities.alias("dissatisfied"))
topics = await df.filter(pl.col("dissatisfied") > 0.5).jev.choice(
    "What is the issue?",
    choices=["billing", "bug", "other"],
    state="review",
)
```

The API and columns match pandas; outputs are native `pl.Series`/`pl.DataFrame`.
Polars uses nulls for missing results; pandas uses NaN for floating outputs and
native nullable string/integer columns. Empty outputs retain their schemas.
LazyFrame expressions are outside v0.

## Controls, errors, and caching

Every method accepts these keyword arguments:

| Option | Default | Behavior |
| --- | --- | --- |
| `client` | `None` | Reuse an official `AsyncTypeSafeClient`; caller owns its lifetime |
| `model` | `None` | Inherit SDK client/environment default (`jev-latest` otherwise) |
| `max_concurrency` | `16` | Maximum simultaneous row evaluations, including retries |
| `cache` | `None` | Opt in with a reusable `MemoryCache` |
| `progress` | `False` | `True` for tqdm, or a synchronous `Progress` callback |
| `errors` | `"raise"` | `"coerce"` returns missing results and an `EvaluationWarning` |
| `nulls` | `"include"` | `"skip"` skips rows with any null context; `"raise"` rejects them |

Null policy applies **after** constructing context, including nested nulls. A row
failure makes every question result for that row missing. `EvaluationError.position`
and `.cause`, or `EvaluationWarning.failures` (`RowFailure.position`/`.cause`), identify
failures without confusing duplicate index labels. Positions refer to the evaluated
frame, starting at zero. Inspect warnings with `warnings.catch_warnings(record=True)`.
Progress snapshots expose `completed`, `total`, `failed`, `skipped`, and `cache_hits`.
Callback exceptions and cancellation propagate, and outstanding workers are drained.

All questions for a row share one request. SDK retries handle transient failures and
backoff (two retries by default); configure its policy and timeouts directly:

```python
from jevframe import MemoryCache
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

cache = MemoryCache(max_entries=10_000)
async with AsyncTypeSafeClient(timeout=30, retry=RetryPolicy(max_retries=3)) as client:
    result = await df.jev.noul(
        "Is this urgent?",
        state="review",
        client=client,
        max_concurrency=8,
        cache=cache,
        progress=True,
    )
```

The memory cache stores only successful, validated complete responses, including
all probabilities. Identical in-flight rows within one evaluation share a request.
Questions, criteria order, context, model, and client/configuration identity affect
cache keys; indexes do not. Supplied clients have separate cache namespaces. Default
clients reuse entries across calls when their environment configuration matches.
There is no disk cache. Model aliases can change: use a pinned model for reproducible
work and `cache.clear()` when fresh results are needed.

No API call runs merely by importing the library. Evaluation sends the chosen
context to TypeSafe. Late context errors can occur after other rows have completed;
completed requests cannot be undone. Credentials are read from the environment or
the SDK client; the library does not discover or load dotenv files.

## marimo example and development

From a checkout:

```sh
# For library development only: uv sync --extra pandas (or --extra polars)
uv sync --extra examples
uv run marimo edit examples/reviews.py
```

Edit the three questions, choose an output layout, and click **Evaluate reviews**
to see the results and probability histograms. The eight synthetic reviews need eight initial
requests, each containing all three questions; repeated evaluations reuse the cache.
The example makes no requests until the button is clicked and a key is configured.

The badge opens the GitHub notebook preview. Fork it into your molab workspace to
add your own secret and run it on a server. The notebook includes inline dependency
metadata that installs `jevframe[pandas]==0.1.0` from PyPI. The badge requires the
notebook to be available on this repository's `main` branch.

For development checks, install all extras:

```sh
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run marimo check --strict examples/reviews.py
uv build
```

Tests use the official SDK with mocked HTTP transport; no API key or live calls
are required. v0 focuses on row-wise decisions, with no chat interface, automatic
analysis, generated code, custom dtypes, Excel integration, or provider framework.
