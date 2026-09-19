import subprocess
import sys
from pathlib import Path


def test_marimo_example_loads_and_evaluates_with_mocked_inference():
    code = r"""
import io
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx2
from fsspec.implementations.http import HTTPFileSystem
from typesafe_sdk import AsyncTypeSafeClient

from examples.reviews import app
from jevframe import _engine

# Keep both the public dataset download and inference offline in this test.
archive_bytes = io.BytesIO()
with zipfile.ZipFile(archive_bytes, 'w') as archive:
    archive.writestr(
        'sentiment labelled sentences/amazon_cells_labelled.txt',
        ''.join(f'Customer review {i}\t{i % 2}\n' for i in range(1000)),
    )
async def download(self, url, destination, **kwargs):
    assert url.startswith('https://archive.ics.uci.edu/static/public/331/')
    assert self.kwargs['timeout'] == 30
    Path(destination).write_bytes(archive_bytes.getvalue())
HTTPFileSystem._get_file = download
async def exists(self, url, **kwargs):
    assert url.startswith('https://archive.ics.uci.edu/static/public/331/')
    return True
HTTPFileSystem._exists = exists

calls = []
def handler(request):
    payload = json.loads(request.content)
    calls.append(payload)
    return httpx2.Response(200, json={
        'model': 'test', 'usage': {},
        'answers': {
            'dissatisfied': {'type': 'noul', 'noul': 0.4},
            'defect': {'type': 'noul', 'noul': 0.7},
            'topic': {
                'type': 'choice', 'choice': 'functionality', 'confidence': 0.8,
                'probabilities': {
                    'functionality': 0.8, 'usability': 0.05, 'value': 0.05,
                    'service': 0.05, 'other': 0.05,
                },
            },
        },
    })

_engine._create_client = lambda **kwargs: AsyncTypeSafeClient(
    api_key='test-key', transport=httpx2.MockTransport(handler),
)
os.environ.pop('TYPESAFE_API_KEY', None)
_, initial = app.run()
assert len(initial['all_reviews']) == 1000
assert len(initial['reviews']) == 50
assert initial['reviews'].index.name == 'review_id'
assert set(initial['all_reviews']['reference_sentiment']) == {'positive', 'negative'}
assert initial['review_files'].protocol == 'zip'
assert initial['review_files'].ls('sentiment labelled sentences', detail=False) == [
    'sentiment labelled sentences/amazon_cells_labelled.txt',
]
assert 'results' not in initial
assert calls == []
controls = {
    'evaluate': SimpleNamespace(value=True),
    'question': SimpleNamespace(value='Is this customer dissatisfied?'),
    'defect_question': SimpleNamespace(value='Does this review report a product defect?'),
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
assert len(calls) == 50
assert all(set(call['questions']) == {'dissatisfied', 'defect', 'topic'} for call in calls)
assert all(set(call['state']) == {'review'} for call in calls)
assert definitions['results'].shape == (50, 12)
assert definitions['results']['evaluation_status'].tolist() == ['ok'] * 50
assert definitions['results']['dissatisfied__probability'].tolist() == [0.4] * 50
assert definitions['results']['defect__probability'].tolist() == [0.7] * 50
assert definitions['results']['topic__label'].tolist() == ['functionality'] * 50
assert definitions['results']['topic__p__usability'].tolist() == [0.05] * 50
assert definitions['decisions'].shape == (50, 9)
assert any(output is not None for output in outputs)
# Repeat using the same data/cache, then edit one question: only the latter needs inference.
controls.update({name: definitions[name] for name in (
    'cache', 'reviews', 'all_reviews', 'csv', 'load_public_reviews',
)})
app.run(defs=controls)
assert len(calls) == 50
controls['output_layout'] = SimpleNamespace(value='struct')
_, packed = app.run(defs=controls)
assert len(calls) == 50
assert packed['results'].shape == (50, 4)
assert packed['decisions'].columns.tolist() == ['result']
assert packed['decisions']['result'].tolist() == definitions['decisions'].to_dict('records')
controls['defect_question'] = SimpleNamespace(value='Is the product broken?')
app.run(defs=controls)
assert len(calls) == 100
assert calls[-1]['questions']['defect']['instructions'] == 'Is the product broken?'
# Empty questions are rejected before spending requests.
controls['question'] = SimpleNamespace(value='  ')
_, empty_question = app.run(defs=controls)
assert 'results' not in empty_question
assert len(calls) == 100
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
