# Echo automation

Echo keeps two workflows active:

| Workflow | Runs on | Purpose |
| --- | --- | --- |
| `echo-ci.yml` | Pull requests, pushes to `main`, manual dispatch | Compile Python sources with Python 3.10 and validate editable-install metadata without installing CUDA/training dependencies. |
| `secrets_scan.yml` | Pull requests, pushes to `main` or `v0.*` | Preserve the existing TruffleHog secret scan. |

These are basic smoke checks. They do not validate training behavior,
dependency compatibility, type coverage, or GPU/NPU execution. Training changes
still need validation in the pinned environment documented in the root README.
The local `.pre-commit-config.yaml` remains available for developers.

## Archived upstream workflows

The 34 inherited verl workflows and their original README are preserved unchanged
in [`.github/legacy-workflows`](../legacy-workflows/). GitHub only discovers
workflows under `.github/workflows`, so these archived definitions do not create
PR checks, scheduled runs, runner cleanup jobs, or autofix PRs.

They depend on upstream infrastructure or files not provided by Echo, including
verl-project runners, model/dataset caches, `requirements-test.txt`, and
`tests/special_sanity`. Some cleanup jobs use `if: always()` even when runner
creation is skipped outside verl-project. The old pre-commit workflow also runs
against every tracked file and reports existing lint/formatting issues on a schedule.

Do not copy an archived workflow back unchanged. Adapt its triggers, runner,
permissions, dependencies, and test paths to Echo before enabling it.

## Dependabot PR policy

[`.github/dependabot.yml`](../dependabot.yml) explicitly pauses automated Python
dependency PRs for the root manifests:

- `open-pull-requests-limit: 0` disables version-update PRs.
- `ignore: [{dependency-name: "*"}]` also pauses security-update PRs. An empty
  `updates` list or a zero version-update limit alone does not configure this
  security-update policy.
- The required monthly schedule does not override these restrictions.

This does not disable Dependabot alerts or the dependency graph in repository
settings. Review vulnerability alerts and update the pinned Python/CUDA stack
manually. Remove the ignore rule to resume automated security-update PRs; raise
the PR limit to resume version-update PRs after reviewing compatibility.

GitHub documents these controls in the
[Dependabot options reference](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference)
and [security update configuration](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/configure-security-updates).

## Applying this cleanup

The policy takes effect when these changes reach the default branch (`main`).
The file changes do not themselves delete existing Dependabot PRs or branches,
and historical failed checks remain in GitHub's run history. Close unwanted
existing bot PRs and delete their branches separately after the policy is active.

Update older development branches from `main` before opening more PRs so they do
not reintroduce the archived workflow files. If required checks are configured
later in branch protection or rulesets, use the active checks instead of retired
verl job names.
