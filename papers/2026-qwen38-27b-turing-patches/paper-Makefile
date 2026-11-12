TECTONIC ?= tectonic
export SOURCE_DATE_EPOCH := 1794506400
export FORCE_SOURCE_DATE := 1
export TZ := UTC
.PHONY: all check
all:
	$(TECTONIC) --only-cached --untrusted --keep-logs main.tex
	$(MAKE) check
check:
	@test -s main.log
	@! grep -Ein 'undefined|multiply defined|Overfull .hbox|Float too large|^!' main.log
