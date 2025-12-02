package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	// Create a shared context
	ctx := context.Background()

	// Run the stages of the pipeline
	if err := Build(ctx); err != nil {
		fmt.Println("Error:", err)
		panic(err)
	}
}

func Build(ctx context.Context) error {
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	// Mount the entire repository at /repo so container layout matches local repo
	repo := client.Host().Directory("../")

	base := client.Container().
		From("python:3.12.2-bookworm").
		WithDirectory("/repo", repo). // mount host repo root at /repo
		WithWorkdir("/repo/source").  // run commands from source folder
		WithExec([]string{"python", "--version"})

	// Install pip requirements from repo root if they exist
	base = base.WithExec([]string{
		"bash", "-lc",
		"if [ -f /repo/requirements.txt ]; then python -m pip install -r /repo/requirements.txt; fi",
	})
	if _, err := base.Stdout(ctx); err != nil {
		return err
	}

	fmt.Println("Initializing data loading")
	data := base.WithExec([]string{"python", "data.py"})
	if _, err := data.Stdout(ctx); err != nil {
		return err
	}

	fmt.Println("Initializing preprocessing")
	preprocess := data.WithExec([]string{"python", "preprocess.py"})
	if _, err := preprocess.Stdout(ctx); err != nil {
		return err
	}

	fmt.Println("Initializing training")
	train := preprocess.WithExec([]string{"python", "train.py"})
	if _, err := train.Stdout(ctx); err != nil {
		return err
	}

	fmt.Println("Initializing evaluation")
	evaluation := train.WithExec([]string{"python", "evaluation.py"})
	if _, err := evaluation.Stdout(ctx); err != nil {
		return err
	}

	// Export artifacts and mlruns from the repo layout
	fmt.Println("Exporing Artifacts and Data")
	_, err = evaluation.
		Directory("/repo/artifacts").
		Export(ctx, "../artifacts")
	if err != nil {
		return err
	}

	_, err = evaluation.
		Directory("/repo/data").
		Export(ctx, "../data")
	if err != nil {
		return err
	}

	fmt.Println("Pipeline complete")
	return nil
}
