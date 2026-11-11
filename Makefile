PYTHON ?= python

.PHONY: inspect clean weather panel eda monthly hypothesis all

inspect:
	$(PYTHON) scripts/01_inspect_raw.py

clean:
	$(PYTHON) scripts/02_clean_trips.py

weather:
	$(PYTHON) scripts/03_fetch_weather.py

panel:
	$(PYTHON) scripts/04_build_panel.py

eda:
	$(PYTHON) scripts/05_eda.py

monthly:
	$(PYTHON) scripts/05b_monthly.py

hypothesis:
	$(PYTHON) scripts/06_validate_hypothesis.py

all: inspect clean weather panel eda monthly hypothesis
