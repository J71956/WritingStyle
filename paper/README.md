# Paper build notes

`main.tex` is the systems/case-study write-up of the test campaign. It compiles
two ways.

## With the ACL template (submission format)

Download the ACL Rolling Review LaTeX template and drop `acl.sty` and
`acl_natbib.bst` into this directory, then:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`main.tex` detects `acl.sty` and switches to ACL style automatically. Overleaf is
the fallback path if you would rather not install a TeX distribution: create a
project from the ACL template and upload `main.tex`, `references.bib` and
`figures/`.

## Without it (draft format)

With no `acl.sty` present the same command builds a plain two-column article, so
the draft always compiles. **This is not camera-ready format** — do not submit
the fallback output.

## Before submitting

1. Run the campaign and the analysis so `figures/fig_ablation.pdf` and
   `figures/fig_by_doc_type.pdf` exist:

```bash
python experiments/run_campaign.py && python experiments/analyze_results.py
```

2. Replace every `\pending{}` in `main.tex`. They render in red, so a `grep -c
   'pending' main.tex` of zero is the check that the draft is finished. Each
   replaced number must trace to a row under `artifacts/eval/`.

3. Read `references.bib` against the published record. It is a starter set
   assembled from memory of the literature and **entries may be inaccurate** —
   verify authors, venues, years and page numbers before citing any of them.
