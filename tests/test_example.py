import subprocess
import sys
from pathlib import Path


def test_marimo_example_loads_and_evaluates_with_mocked_inference():
    code = r"""
import io
import json
import os
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx2
from fsspec.implementations.http import HTTPFileSystem
from typesafe_sdk import AsyncTypeSafeClient

from examples.reviews import app
from jevframe import EvaluationWarning, MemoryCache

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
destinations = []
fail_reviews = set()
gateway_limited = False
personal_unauthorized = False
def handler(request):
    assert request.method == 'POST'
    assert request.url.path == '/v1/systemone'
    payload = json.loads(request.content)
    calls.append(payload)
    destinations.append((request.url.host, request.headers['authorization']))
    if gateway_limited and request.url.host == 'jev-api-wrapper-production.up.railway.app':
        return httpx2.Response(429, json={'error': {'code': 'daily_limit'}},
                               headers={'Retry-After': '3600'})
    if personal_unauthorized and request.url.host == 'personal-typesafe.test':
        return httpx2.Response(401, json={'message': 'Invalid personal key'})
    if payload['state']['review'] in fail_reviews:
        return httpx2.Response(400, json={'message': 'Test failure'})
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

# Intercept the public SDK for both the demo's explicit client and jevframe's
# owned client. Keep the official class so client type checks still run.
original_init = AsyncTypeSafeClient.__init__
original_close = AsyncTypeSafeClient.aclose
configurations, clients, closed_clients = [], [], []
def mock_init(self, *args, **kwargs):
    configurations.append(dict(kwargs))
    kwargs['transport'] = httpx2.MockTransport(handler)
    original_init(self, *args, **kwargs)
    clients.append(self)
async def tracked_close(self):
    await original_close(self)
    closed_clients.append(self)
AsyncTypeSafeClient.__init__ = mock_init
AsyncTypeSafeClient.aclose = tracked_close
os.environ.pop('TYPESAFE_API_KEY', None)
os.environ['TYPESAFE_DEFAULT_MODEL'] = 'test-model'
os.environ['TYPESAFE_BASE_URL'] = 'https://personal-typesafe.test'
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
assert initial['using_demo_gateway'] is True
assert initial['request_preview']['model'] == 'jev-latest'
assert initial['request_preview']['state'] == {
    'review': initial['reviews'].iloc[0]['review'],
}
assert list(initial['request_preview']['questions']) == ['dissatisfied', 'defect', 'topic']
controls = {
    'evaluate': SimpleNamespace(value=True),
    'question': SimpleNamespace(value='Is this customer dissatisfied?'),
    'defect_question': SimpleNamespace(value='Does this review report a product defect?'),
    'topic_question': SimpleNamespace(value='What is the main issue?'),
    'output_layout': SimpleNamespace(value='columns'),
}
# A missing (or blank) secret selects the sponsored gateway, without changing
# SDK environment settings. Its supported model overrides personal model defaults.
os.environ['TYPESAFE_API_KEY'] = '   '
_, missing_key = app.run(defs=controls)
assert len(calls) == 50
assert missing_key['results']['evaluation_status'].tolist() == ['ok'] * 50
assert missing_key['using_demo_gateway'] is True
assert missing_key['evaluation_requests'] == calls
assert all(call['model'] == 'jev-latest' for call in calls)
assert set(destinations) == {
    ('jev-api-wrapper-production.up.railway.app', 'Bearer public-demo'),
}
assert configurations[-1]['retry'].max_retries == 0
assert clients == closed_clients
assert os.environ['TYPESAFE_API_KEY'] == '   '
assert os.environ['TYPESAFE_BASE_URL'] == 'https://personal-typesafe.test'
# Quota failures stay missing, do not retry Retry-After, and close the demo client.
gateway_limited = True
calls.clear()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', EvaluationWarning)
    _, limited = app.run(defs={**controls, 'reviews': missing_key['reviews'].head(3)})
assert len(calls) == 3
assert limited['decisions'].isna().all().all()
assert limited['evaluation_progress'].failed == 3
assert any(isinstance(warning.message, EvaluationWarning) for warning in caught)
assert clients == closed_clients
gateway_limited = False
calls.clear()
destinations.clear()
# Secrets added after opening the notebook work on the next click.
os.environ['TYPESAFE_API_KEY'] = ' test-key '
outputs, definitions = app.run(defs=controls)
assert len(calls) == 50
assert definitions['using_demo_gateway'] is False
assert definitions['request_model'] == 'test-model'
assert set(destinations) == {('personal-typesafe.test', 'Bearer test-key')}
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
# Every displayed body equals the corresponding real SDK HTTP body, including
# the environment-selected model and official question serialization.
assert definitions['evaluation_requests'] == calls
assert definitions['request_preview'] == calls[0]
assert definitions['inspected_request'] == calls[0]
assert definitions['inspected_fields'] == definitions['decisions'].iloc[0].to_dict()
assert definitions['evaluation_progress'].completed == 50
assert definitions['evaluation_progress'].cache_hits == 0
assert definitions['evaluation_progress'].failed == 0
# Repeat using the same data/cache, then edit one question: only the latter needs inference.
controls.update({name: definitions[name] for name in (
    'cache', 'reviews', 'all_reviews', 'csv', 'load_public_reviews',
)})
_, repeated = app.run(defs=controls)
assert len(calls) == 50
assert repeated['evaluation_progress'].cache_hits == 50
# Select by position, preserving the original shuffled dataframe index.
controls['request_row'] = SimpleNamespace(value=7)
_, selected = app.run(defs=controls)
assert len(calls) == 50
assert selected['request_preview'] == calls[7]
assert selected['inspected_request'] == calls[7]
assert selected['inspected_fields'] == definitions['decisions'].iloc[7].to_dict()
controls['output_layout'] = SimpleNamespace(value='struct')
_, packed = app.run(defs=controls)
assert len(calls) == 50
assert packed['results'].shape == (50, 4)
assert packed['decisions'].columns.tolist() == ['result']
assert packed['decisions']['result'].tolist() == definitions['decisions'].to_dict('records')
assert packed['inspected_fields'] == selected['inspected_fields']
assert packed['evaluation_progress'].cache_hits == 50
controls['defect_question'] = SimpleNamespace(value='Is the product broken?')
_, changed = app.run(defs=controls)
assert len(calls) == 100
assert calls[-1]['questions']['defect']['instructions'] == 'Is the product broken?'
assert changed['evaluation_requests'] == calls[50:]
assert changed['inspected_request'] == calls[57]
assert definitions['evaluation_requests'][7]['questions']['defect']['instructions'] == (
    'Does this review report a product defect?'
)
# Empty questions are rejected before spending requests.
controls['question'] = SimpleNamespace(value='  ')
_, empty_question = app.run(defs=controls)
assert 'results' not in empty_question
assert len(calls) == 100
# Inspecting a failed row must not render invented probabilities in either layout.
controls['question'] = SimpleNamespace(value='Is this customer dissatisfied?')
fail_reviews.add(definitions['reviews'].iloc[7]['review'])
for layout in ('columns', 'struct'):
    controls['cache'] = MemoryCache()
    controls['output_layout'] = SimpleNamespace(value=layout)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', EvaluationWarning)
        _, failed = app.run(defs=controls)
    assert any(isinstance(warning.message, EvaluationWarning) for warning in caught)
    assert failed['inspected_fields'] is None
    assert failed['results'].iloc[7]['evaluation_status'] == 'failed'
    assert failed['evaluation_progress'].failed == 1
# A configured but invalid personal key never silently falls back to sponsorship.
personal_unauthorized = True
calls.clear()
destinations.clear()
with warnings.catch_warnings(record=True):
    warnings.simplefilter('always', EvaluationWarning)
    _, unauthorized = app.run(defs={
        **controls, 'cache': MemoryCache(),
        'reviews': definitions['reviews'].head(2),
        'request_row': SimpleNamespace(value=0), 'output_layout': SimpleNamespace(value='columns'),
    })
assert len(calls) == 2
assert set(destinations) == {('personal-typesafe.test', 'Bearer test-key')}
assert unauthorized['decisions'].isna().all().all()
assert clients == closed_clients
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
