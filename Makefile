FOLDERS := transformers_implementation

format: 
	isort --profile black "$(FOLDERS)"
	black "$(FOLDERS)"

lint:
	flake8 "$(FOLDERS)"