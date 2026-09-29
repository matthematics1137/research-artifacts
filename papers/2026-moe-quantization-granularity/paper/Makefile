TECTONIC ?= tectonic
TECTONIC_FLAGS ?= --untrusted --only-cached --color never
SOURCE_DATE_EPOCH ?= 1788191483
export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE := 1
export TZ := UTC

all: main.pdf

main.pdf: main.tex references.bib derived-p4r1/claims.tex figures/p4r1/fig_matched.pdf figures/p4r1/fig_delta.pdf
	$(TECTONIC) $(TECTONIC_FLAGS) main.tex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out

.PHONY: all clean
