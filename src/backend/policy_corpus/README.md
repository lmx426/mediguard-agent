# Policy Corpus

This directory stores the first-stage source corpus for the policy RAG pilot.
It is intentionally separated from runtime fixtures and model assets.

The corpus is evidence material for future Case Agent policy lookup. It must
not be used to automatically reject, approve, punish, or change a case result.
Human auditors remain the final decision makers.

## Scope

The first pilot focuses on these closed-loop audit questions:

- Beijing insured person, Shanghai treatment, cross-region outpatient or
  emergency manual reimbursement.
- Emergency status, filing or emergency exception material verification.
- Local treatment catalogue versus insured-place reimbursement policy.
- Drug catalogue and limited-payment condition lookup.
- CT and imaging fee audit signals: repeated charges, unbundling, over-standard
  charges, and item-name normalization.
- Local outpatient audit signals: high inspection ratio, high drug ratio,
  missing registration evidence, high-frequency multi-institution visits.

## Directory Layout

```text
src/backend/policy_corpus/
  seeds/
    p0_sources.yaml              # Official seeds and crawl policy.
  raw/
    national/
    beijing/
    shanghai/
    attachments/                 # Downloaded PDFs, spreadsheets, images, zips.
  manifest/
    policy_source_manifest.jsonl # One line per crawled source.
  clean/
    markdown/                    # Deterministic Markdown converted from raw HTML.
  rag_ready/                     # Accepted corpus that may be ingested by RAG.
  chunks/
    policy_chunks.jsonl          # Future chunk output, not produced in stage 1.
  reports/                       # Future coverage and validation reports.
```

Stage 1 only produces `raw`, `manifest`, and optionally `clean/markdown`.
Do not run embedding or vector-store ingestion until raw source traceability is
stable.

Only sources rated as `basic_satisfied` or `fully_satisfied` may be copied or
transformed into `rag_ready/`. Partially satisfied sources, raw-only attachments,
unparsed PDFs or spreadsheets, reference-only typical cases, and noise pages
must stay outside the RAG-ready corpus until they pass acceptance.

Some official sources are direct PDF, WPS, Word, Excel, CSV, or ZIP files rather
than HTML pages. They are saved as raw source records with `clean_path = null`
and are not parsed during stage 1.

## Stage 1 Cleaning Rules

Raw source files are immutable once written. The clean Markdown layer may:

- remove navigation, footer, sharing widgets, ads, script/style noise, and empty
  blocks;
- preserve titles, headings, tables, links, source URL, and attachment links;
- add front matter with jurisdiction, policy domain, crawl timestamp, and source
  ID;
- normalize whitespace and repeated blank lines;
- mark publish/effective/expiry dates as `unknown` when not deterministically
  found.

The crawler must not use LLM rewriting or LLM filtering in this stage.
Noise pruning is deterministic: non-seed pages whose titles match configured
exclude keywords can be removed from manifest, raw HTML, and clean Markdown.

## Traceability Fields

Every source record should keep:

- `source_id`
- `title`
- `jurisdiction`
- `policy_domain`
- `url`
- `issuing_authority`
- `publish_date`
- `effective_date`
- `expiry_date`
- `status`
- `version_note`
- `supersedes`
- `doc_type`
- `legal_weight`
- `case_tags`
- `raw_path`
- `clean_path`
- `sha256`
- `fetched_at`
- `attachments`

Unknown version fields are allowed, but the uncertainty must be explicit.

## Git Policy

Text, Markdown, YAML, JSONL manifests, and small metadata reports can be stored
directly in Git. Large raw attachments under `raw/` should be tracked by Git LFS
according to the repository `.gitattributes`.
