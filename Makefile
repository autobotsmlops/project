install: 
	@echo "Installing..."
	@python.exe -m pip install --upgrade pip
	@pip install -r requirements.txt
	@echo "Done."

lint:
	@echo "Linting using pylint..."
	@pylint --fail-under=8 --exit-zero --max-line-length=120 src 
	@echo "Done."

run:
	@echo "Running Nothing..."
	@python src/main.py
	@echo "Done."