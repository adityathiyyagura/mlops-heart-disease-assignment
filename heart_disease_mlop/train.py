from heart_disease_mlop.modeling import train_and_track


if __name__ == "__main__":
    result = train_and_track()
    print(f"Best model: {result['best_model_name']}")
    for model_name, model_result in result["model_results"].items():
        print(f"\n{model_name} best parameters: {model_result['best_params']}")
        print(f"{model_name} cross-validation: {model_result['cross_validation']}")
        print(f"{model_name} held-out test: {model_result['held_out_test']}")
