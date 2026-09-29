TECTONIC ?= $(HOME)/.local/bin/tectonic
.PHONY: all invalidated main.pdf figures dark explainer explainer.pdf explainer-dark brief brief.pdf brief-dark historical-forensic forensic_main.pdf forensic_explainer.pdf forensic_brief.pdf forensic-dark forensic-explainer-dark forensic-brief-dark forensic-figures audit audit-check figure-check p4r1-figures p4r1-manuscript-check p4r1-previews p4r1-preparation-check p4r1-canonical-candidate p4r1-final-reading

P4R1_DERIVED ?= derived-p4r1
P4R1_CLAIMS_SHA256 ?=
P4R1_NARRATIVE_REVIEW ?=
P4R1_NARRATIVE_REVIEW_SHA256 ?=
P4R1_RELEASE_MANIFEST ?= publication/papers/moe-quantization-granularity/release-manifest.json

all: p4r1-canonical-candidate

# The historical six-cell campaign failed the post-audit endpoint-provenance
# standard. An ordinary build must never make its PDF look canonical.
invalidated:
	@echo "Paper 4 is publication-blocked: historical primary evidence is forensic only; complete P4R1 and its narrative review." >&2
	@false

# The historical TeX sources were not frozen as a separate source tree.  Never
# compile today's P4R1 sources under a forensic filename: that would mislabel
# current results as the invalidated historical manuscript.  The already
# archived forensic PDFs remain available under forensic-historical/.
historical-forensic forensic_main.pdf forensic_explainer.pdf forensic_brief.pdf forensic-dark forensic-explainer-dark forensic-brief-dark:
	@echo "Historical rebuild disabled: no frozen historical TeX source tree exists; inspect forensic-historical/ instead." >&2
	@false

# The 2026-09-05 author direction allows a fully prepared candidate before
# the one final read. These targets never approve content or publish it.
main.pdf: p4r1-canonical-candidate
figures: p4r1-figures
explainer explainer.pdf brief brief.pdf: p4r1-final-reading
dark explainer-dark brief-dark: invalidated

audit:
	python3 scripts/check_historical_snapshot.py

audit-check:
	python3 scripts/check_historical_snapshot.py
	python3 scripts/test_verify_claims.py
	python3 scripts/test_approve_p4r1_narrative.py
	python3 scripts/test_check_p4r1_manuscript.py

# Supplementary analysis-only figures consume derived/claims.json exclusively.
# They are diagnostic audit views and are intentionally absent from main.tex.
figure-check:
	python3 scripts/test_make_figures.py

forensic-figures: figure-check derived/claims.json
	python3 scripts/make_figures.py --output-dir figures/forensic-historical
	python3 scripts/make_figures.py --dark --output-dir figures/forensic-historical/dark
# Post-audit preview path. It never replaces the historical derived tree or
# canonical PDFs; release promotion remains a separate reviewed operation.
p4r1-figures:
	test -f $(P4R1_DERIVED)/claims.json
	python3 scripts/make_figures.py --claims $(P4R1_DERIVED)/claims.json --output-dir figures/p4r1
	python3 scripts/make_figures.py --dark --claims $(P4R1_DERIVED)/claims.json --output-dir figures/p4r1/dark

p4r1-manuscript-check:
	test -n "$(P4R1_CLAIMS_SHA256)"
	test -n "$(P4R1_NARRATIVE_REVIEW)"
	test -n "$(P4R1_NARRATIVE_REVIEW_SHA256)"
	python3 scripts/check_p4r1_manuscript.py \
		--claims $(P4R1_DERIVED)/claims.json \
		--claims-sha256 $(P4R1_CLAIMS_SHA256) \
		--narrative-review $(P4R1_NARRATIVE_REVIEW) \
		--narrative-review-sha256 $(P4R1_NARRATIVE_REVIEW_SHA256)

p4r1-previews: p4r1-figures p4r1-manuscript-check
	test -f $(P4R1_DERIVED)/claims.tex
	printf '%s\n%s\n' '\def\PfourClaimsPath{$(P4R1_DERIVED)/claims}' '\input{main.tex}' > p4r1_main.tex
	$(TECTONIC) p4r1_main.tex
	printf '%s\n%s\n%s\n' '\def\DARKMODE{}' '\def\PfourClaimsPath{$(P4R1_DERIVED)/claims}' '\input{main.tex}' > p4r1_main_dark.tex
	$(TECTONIC) p4r1_main_dark.tex
	printf '%s\n%s\n' '\def\PfourClaimsPath{$(P4R1_DERIVED)/claims}' '\input{explainer.tex}' > p4r1_explainer.tex
	$(TECTONIC) p4r1_explainer.tex
	printf '%s\n%s\n%s\n' '\def\DARKMODE{}' '\def\PfourClaimsPath{$(P4R1_DERIVED)/claims}' '\input{explainer.tex}' > p4r1_explainer_dark.tex
	$(TECTONIC) p4r1_explainer_dark.tex
	printf '%s\n%s\n' '\def\PfourClaimsPath{$(P4R1_DERIVED)/claims}' '\input{brief.tex}' > p4r1_brief.tex
	$(TECTONIC) p4r1_brief.tex
	printf '%s\n%s\n%s\n' '\def\DARKMODE{}' '\def\PfourClaimsPath{$(P4R1_DERIVED)/claims}' '\input{brief.tex}' > p4r1_brief_dark.tex
	$(TECTONIC) p4r1_brief_dark.tex
	rm -f p4r1_main.tex p4r1_main_dark.tex p4r1_explainer.tex p4r1_explainer_dark.tex p4r1_brief.tex p4r1_brief_dark.tex

# The historical narrative-attestation target above is retained for old
# workflows. It is not a prerequisite under the scoped preparation-only
# deviation recorded in PREPARATION_WORKFLOW_2026-09-05.md. Receipt-independent
# scientific checks remain required; real content approval is still mandatory
# before any publication plan can be applied.
p4r1-preparation-check:
	python3 scripts/build_review_copies.py --self-test
	python3 scripts/test_bibliography_contract.py

p4r1-canonical-candidate: p4r1-preparation-check
	cd .. && python3 publication/releasectl.py build-paper \
		$(P4R1_RELEASE_MANIFEST) --repo-root . --json

p4r1-final-reading: p4r1-canonical-candidate
	python3 scripts/build_final_read.py
