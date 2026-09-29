TECTONIC ?= $(HOME)/.local/bin/tectonic
.PHONY: all release review audit audit-check figures light dark release-gate \
	pre-figure-gate \
	public-allowlist review-candidates forensic forensic-dark clean-wrappers

# Canonical outputs are deliberately all-or-nothing: P2R1 promotion,
# result-specific narrative review, and the privacy/statistics checks must pass
# before any light or dark review PDF is rebuilt.
all: release

release: audit-check main.pdf main_dark.pdf explainer.pdf explainer_dark.pdf

review: release

light: audit-check main.pdf explainer.pdf

dark: audit-check main_dark.pdf explainer_dark.pdf

pre-figure-gate:
	python3 scripts/check_release_state.py --pre-figure

release-gate: pre-figure-gate
	python3 scripts/check_release_state.py

audit:
	python3 scripts/build_claims.py

audit-check: release-gate
	python3 scripts/build_claims.py --check
	python3 scripts/test_claims.py

figures: pre-figure-gate derived/claims.json
	python3 scripts/make_figures.py
	python3 scripts/make_figures.py --dark

public-allowlist:
	test -n "$(P2_PUBLIC_RELEASE_ROOT)"
	python3 scripts/build_public_allowlist.py --release-root "$(P2_PUBLIC_RELEASE_ROOT)"

# These unmistakably noncanonical PDFs are the artifacts the author reviews.
# The public allowlist and narrative receipt bind their exact hashes. Canonical
# targets below only copy these already-scanned bytes; they never recompile a
# different PDF after the privacy/editorial audit.
review-candidates: figures
	python3 scripts/check_release_state.py --candidate-sources
	mkdir -p review/light review/dark
	$(TECTONIC) --outdir review/light main.tex
	$(TECTONIC) --outdir review/light explainer.tex
	printf '%s\n%s\n' '\def\DARKMODE{}' '\input{main.tex}' > main_dark_review.tex
	$(TECTONIC) --outdir review/dark main_dark_review.tex
	rm -f main_dark_review.tex
	printf '%s\n%s\n' '\def\DARKMODE{}' '\input{explainer.tex}' > explainer_dark_review.tex
	$(TECTONIC) --outdir review/dark explainer_dark_review.tex
	rm -f explainer_dark_review.tex

main.pdf: audit-check review/light/main.pdf
	cp --preserve=mode,timestamps review/light/main.pdf main.pdf
	cmp review/light/main.pdf main.pdf

main_dark.pdf: audit-check review/dark/main_dark_review.pdf
	cp --preserve=mode,timestamps review/dark/main_dark_review.pdf main_dark.pdf
	cmp review/dark/main_dark_review.pdf main_dark.pdf

explainer.pdf: audit-check review/light/explainer.pdf
	cp --preserve=mode,timestamps review/light/explainer.pdf explainer.pdf
	cmp review/light/explainer.pdf explainer.pdf

explainer_dark.pdf: audit-check review/dark/explainer_dark_review.pdf
	cp --preserve=mode,timestamps review/dark/explainer_dark_review.pdf explainer_dark.pdf
	cmp review/dark/explainer_dark_review.pdf explainer_dark.pdf

# Forensic outputs have unmistakable filenames and never overwrite a
# canonical main/explainer PDF while the historical draft remains invalid.
forensic:
	python3 scripts/check_release_state.py --forensic-invalidated-build
	test -f forensic/p2g-invalidated/main.pdf
	@echo "Preserved invalid P2G PDF: forensic/p2g-invalidated/main.pdf"

forensic-dark:
	python3 scripts/check_release_state.py --forensic-invalidated-build
	test -f forensic/p2g-invalidated/main_dark.pdf
	@echo "Preserved invalid P2G dark PDF: forensic/p2g-invalidated/main_dark.pdf"

clean-wrappers:
	rm -f main_dark.tex explainer_dark.tex main_dark_review.tex \
		explainer_dark_review.tex forensic_*_invalidated*.tex
