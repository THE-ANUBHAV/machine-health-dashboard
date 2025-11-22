import pickle

models = [
    "random_forest.pkl",
    "gradient_boosting.pkl"
]

for path in models:
    print(f"\n--- Inspecting {path} ---\n")
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)

        print("Model type:", type(model).__name__)
        print("n_features_in:", getattr(model, "n_features_in_", None))

        if hasattr(model, "feature_names_in_"):
            print("feature_names_in_:", list(model.feature_names_in_))
        else:
            print("feature_names_in_: NOT AVAILABLE")

    except Exception as e:
        print("Failed:", e)
