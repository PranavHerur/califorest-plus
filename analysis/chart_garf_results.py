import pickle
import os


def load_random_forest_model(model_path):
    """
    Load a pickled random forest model from the specified path.

    Args:
        model_path (str): Path to the pickled random forest model file

    Returns:
        The unpickled random forest model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Example usage
path = "garf_results/mort_hosp_5000_subjects_20250423_180209.pkl"
rf_model = load_random_forest_model(path)
print(rf_model)

# Get the feature importances
feature_importances = rf_model.feature_importances_

# Print the feature importances
print(feature_importances)
