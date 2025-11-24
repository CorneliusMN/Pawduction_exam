# make runs setup automatically
all: setup 

DIRS = \
	artifacts \
	mlruns \
	mlruns/.trash

setup:
	@echo "Creating necessary directories..."
	@mkdir -p $(DIRS)
	@echo "Setup complete."

.PHONY: all setup
