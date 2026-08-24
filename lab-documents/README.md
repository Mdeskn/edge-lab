# Lab documents

Student-facing handouts for Lab 3, IoT & Edge Computing.

| File | Audience | Purpose |
|---|---|---|
| `Lab_3_EdgeLab_Instructions.docx` | Students | Setting up and starting the app on the lab Raspberry Pi |
| `Lab_3_EdgeLab_Tasks.docx` | Students | The graded task sheet: baselines, strategy, iteration, submission |

## Why these are not in git

The `.docx` files are gitignored. Word rewrites the entire binary on every
save, so committing them adds hundreds of kilobytes per edit to produce a diff
nobody can read. This repository's history is already oversized from committed
binaries.

Only this README is tracked, so the repository still records what belongs here.

**Consequence worth knowing:** these files are therefore not backed up by git.
Keep the canonical copy in OneDrive or on ILIAS, and treat the copy in this
folder as a working copy.

## They must stay in sync with the code

Both documents contain code students copy verbatim. When a document drifts
from the real API the failure is quiet rather than loud, because students are
correctly taught to read metrics with `.get(key, default)`: a wrong key name
returns its default instead of raising, so the strategy runs, the dashboard
updates, and a score comes out having ignored that signal entirely.

This has already happened twice:

- The Instructions taught `def decide(self, metrics)`. The real method takes no
  argument and reads `self.gpu_metrics`.
- The Tasks sheet taught `gpu_utilization_percent` and `memory_used_mb`. The
  real keys are `gpu_util_pct` and `gpu_mem_used_mb`. The sheet's own worked
  example tested `gpu_util > 70`, which could therefore never be true: it
  returned `"remote"` while the GPU was saturated, the wrong call in the phase
  the lab is built around, and every check still passed.

`client/tests/test_student_api.py` now guards the API surface these documents
teach: it asserts the metric keys exist, that `decide()` takes no arguments,
and that the README skeleton would import correctly inside the container. Those
tests cannot read the `.docx` files, so **after editing a handout, re-check the
code snippets by hand against `client/student/sp_agent_base.py`.**

## Verifying a snippet

Any strategy shown in a handout should survive the submission checker. Paste it
into `client/student/sp_agent.py` and run:

```bash
python scripts/check_sp_agent.py
```

It loads the agent and calls `decide()` against five synthetic metric
snapshots, including the cold-start case where no Kafka message has arrived
yet, and it rejects imports that resolve from the repository root but not
inside the container. Restore the template afterwards:

```bash
git checkout client/student/sp_agent.py
```

## Extracting the text

To read or diff a handout without opening Word:

```bash
python3 -c "
import zipfile, re, sys
x = zipfile.ZipFile(sys.argv[1]).read('word/document.xml').decode('utf-8','ignore')
x = re.sub(r'</w:p>', chr(10), x)
t = re.sub(r'<[^>]+>', '', x)
for a, b in (('&amp;','&'), ('&lt;','<'), ('&gt;','>'), ('&quot;','\"')):
    t = t.replace(a, b)
print(chr(10).join(l for l in t.split(chr(10)) if l.strip()))
" Lab_3_EdgeLab_Tasks.docx
```
