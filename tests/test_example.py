import subprocess
import sys
from pathlib import Path


def test_marimo_example_loads_and_evaluates_with_mocked_inference():
    code = """
import json
import os
from types import SimpleNamespace

import httpx2
from typesafe_sdk import AsyncTypeSafeClient

from examples.reviews import app
from jevframe import _engine

calls = []
def handler(request):
    payload = json.loads(request.content)
    calls.append(payload)
    return httpx2.Response(200, json={
        'model': 'test', 'usage': {},
        'answers': {
            'dissatisfied': {'type': 'noul', 'noul': 0.4},
            'urgent': {'type': 'noul', 'noul': 0.7},
            'topic': {
                'type': 'choice', 'choice': 'billing', 'confidence': 0.8,
                'probabilities': {'billing': 0.8, 'bug': 0.1, 'other': 0.1},
            },
        },
    })

_engine._create_client = lambda **kwargs: AsyncTypeSafeClient(
    api_key='test-key', transport=httpx2.MockTransport(handler),
)
os.environ.pop('TYPESAFE_API_KEY', None)
_, initial = app.run()
assert len(initial['reviews']) == 8
assert 'results' not in initial
assert calls == []
controls = {
    'evaluate': SimpleNamespace(value=True),
    'question': SimpleNamespace(value='Is this customer dissatisfied?'),
    'urgency_question': SimpleNamespace(value='Does this customer need urgent help?'),
    'topic_question': SimpleNamespace(value='What is the main issue?'),
    'output_layout': SimpleNamespace(value='columns'),
}
# A click before configuring a secret displays instructions, with no request.
_, missing_key = app.run(defs=controls)
assert 'results' not in missing_key
assert calls == []
# Secrets added after opening the notebook work on the next click.
os.environ['TYPESAFE_API_KEY'] = 'test-key'
outputs, definitions = app.run(defs=controls)
assert len(calls) == 8
assert all(set(call['questions']) == {'dissatisfied', 'urgent', 'topic'} for call in calls)
assert definitions['results'].shape == (8, 9)
assert definitions['results']['dissatisfied__probability'].tolist() == [0.4] * 8
assert definitions['results']['urgent__probability'].tolist() == [0.7] * 8
assert definitions['results']['topic__label'].tolist() == ['billing'] * 8
assert definitions['results']['topic__p__bug'].tolist() == [0.1] * 8
assert definitions['decisions'].shape == (8, 7)
assert any(output is not None for output in outputs)
# Repeat using the same data/cache, then edit one question: only the latter needs inference.
controls.update(cache=definitions['cache'], reviews=definitions['reviews'])
app.run(defs=controls)
assert len(calls) == 8
controls['output_layout'] = SimpleNamespace(value='struct')
_, packed = app.run(defs=controls)
assert len(calls) == 8
assert packed['results'].shape == (8, 3)
assert packed['decisions'].columns.tolist() == ['result']
assert packed['decisions']['result'].tolist() == definitions['decisions'].to_dict('records')
controls['urgency_question'] = SimpleNamespace(value='Does this require a response today?')
app.run(defs=controls)
assert len(calls) == 16
assert calls[-1]['questions']['urgent']['instructions'] == 'Does this require a response today?'
# Empty questions are rejected before spending requests.
controls['question'] = SimpleNamespace(value='  ')
_, empty_question = app.run(defs=controls)
assert 'results' not in empty_question
assert len(calls) == 16
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
