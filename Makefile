# Make runs setup automatically
all: setup 

DIRS = \
	artifacts \
	data \
	mlruns \
	mlruns/.trash

setup:
	@echo "Creating necessary directories..."
	@mkdir -p $(DIRS)
	@echo "Setup complete."

cleanup:
	@echo "Cleaning up generated directories..."
	@rm -rf $(DIRS)
	@echo "Cleanup complete."

# Prevents name conflict with files, ensures that commands always run
.PHONY: all setup cleanup