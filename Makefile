install: 
	@echo "Installing..."
	@python.exe -m pip install --upgrade pip
	@pip install -r requirements.txt
	@echo "Done."

lint:
	@echo "Linting using pylint..."
	@pylint --fail-under=8 --exit-zero --max-line-length=120 src 
	@echo "Done."

run_flask:
	@echo "Running Flask App..."
	@python app/app.py
	@echo "Done."