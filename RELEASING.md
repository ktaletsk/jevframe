# Releasing jevframe

Releases use GitHub Actions and PyPI trusted publishing. No PyPI API token is stored
in the repository. `.github/workflows/publish.yml` tests and builds distributions in
one job, then publishes those artifacts from a separate job with an OIDC identity.

## One-time PyPI setup

For the first release, add a pending GitHub publisher at
[PyPI account → Publishing](https://pypi.org/manage/account/publishing/):

| Field | Value |
| --- | --- |
| PyPI project name | `jevframe` |
| Owner | `ktaletsk` |
| Repository name | `jevframe` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

The workflow name is the filename, without `.github/workflows/`. Create the `pypi`
environment in the GitHub repository settings and allow deployment from `v*` tags.
PyPI creates the project when this publisher first uploads a release. See
[PyPI's pending-publisher guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Release

1. Update the version in `pyproject.toml` and the pinned dependency in
   `examples/reviews.py`, update `uv.lock`, and commit the changes.
2. Push the commit and a matching version tag, such as `v0.1.0`. The workflow checks
   that the tag matches the package version, runs tests and notebook validation,
   builds and validates the wheel and source distribution, then uploads to PyPI.
3. Verify installation outside this checkout:

   ```sh
   uv run --isolated --no-project --with 'jevframe[pandas,polars]==0.1.0' \
     python -c 'import jevframe.pandas, jevframe.polars'
   ```

If trusted publishing wasn't configured when the workflow ran, finish the PyPI
setup and rerun the failed publish job. For an interrupted upload, retain the
original distribution artifacts and rerun that job; uv skips identical files
already uploaded. Never replace an existing release with rebuilt, different files.
