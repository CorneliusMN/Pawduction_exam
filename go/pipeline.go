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

	python := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("source", client.Host().Directory("../source")).
		WithDirectory("data", client.Host().Directory("../data")).
		WithDirectory("artifacts", client.Host().Directory("../artifacts")).
		WithDirectory("root", client.Host().Directory("../")).
		WithExec([]string{"python", "--version"})

	fmt.Println("Downloading Requirements.txt")
	python = python.WithExec([]string{
		"bash", "-lc",
		"if [ -f /source/requirements.txt ]; then pip install -r /source/requirements.txt; fi",
	})

	fmt.Println("Initializing data loading")
	data := python.WithExec([]string{"python", "source/data.py"})

	fmt.Println("Initializing preprocessing")
	preprocess := data.WithExec([]string{"python", "source/preprocess.py"})

	fmt.Println("Initializing training")
	train := preprocess.WithExec([]string{"python", "source/train.py"})

	fmt.Println("Initializing evaluation")
	evaluation := train.WithExec([]string{"python", "source/evaluation.py"})

	_, err = evaluation.
		Directory("artifacts").
		Export(ctx, "artifacts")
	if err != nil {
		return err
	}

	_, err = evaluation.
		Directory("mlruns").
		Export(ctx, "mlruns")
	if err != nil {
		return err
	}

	fmt.Println("Pipeline complete")
	return nil
}
