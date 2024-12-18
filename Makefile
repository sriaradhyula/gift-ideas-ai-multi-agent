.PHONY: run

install-requirements:
	@source venv/bin/activate && pip install -r requirements.txt

venv:
	python -m venv venv

run: install-requirements
	@if [ ! -d "venv" ]; then $(MAKE) venv; fi
	@source .env && python app/src/gift-ideas-ai-multi-agent.py
