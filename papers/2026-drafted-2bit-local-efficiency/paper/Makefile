TECTONIC ?= tectonic
TECTONIC_FLAGS ?= --untrusted --only-cached --color never
SOURCE_DATE_EPOCH ?= 1789213816
export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE := 1
export TZ := UTC

all: main.pdf

main.pdf: main.tex references.bib
	$(TECTONIC) $(TECTONIC_FLAGS) main.tex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out

.PHONY: all clean
