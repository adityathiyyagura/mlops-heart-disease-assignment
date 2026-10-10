from heart_disease_mlop.modeling import train_and_track


if __name__ == "__main__":
    result = train_and_track()
    print(f"Best model: {result['best_model_name']}")
    print(f"Metrics: {result['metrics']}")
