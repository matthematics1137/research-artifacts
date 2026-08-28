TECTONIC ?= tectonic
TECTONIC_FLAGS ?= --untrusted --only-cached --color never
SOURCE_DATE_EPOCH ?= 1787884410
export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE := 1
export TZ := UTC

all: main.pdf

figures:
	python3 scripts/make_figures.py

main.pdf: main.tex references.bib figures/fig_ladder.pdf figures/fig_offload.pdf figures/fig_artifact.pdf figures/fig_frontier.pdf
	$(TECTONIC) $(TECTONIC_FLAGS) main.tex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out

.PHONY: all figures clean
