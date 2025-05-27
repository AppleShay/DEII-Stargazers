from flask import Flask, request, render_template
import numpy as np
from workerA import get_predictions  # Import the Celery task

app = Flask(__name__)

@app.route('/')
def index():
    # Render the input form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        repos = []

        for i in range(5):
            # Get repository name from the form
            name = request.form.get(f'name{i}', f'Repo {i}')

            # Helper to safely parse float values
            def parse_float(value):
                try:
                    return float(value)
                except:
                    return 0.0

            # Extract and parse feature values for each repo
            row = [
                parse_float(request.form.get(f"issues{i}")),
                parse_float(request.form.get(f"size_kb{i}")),
                parse_float(request.form.get(f"topics{i}")),
                parse_float(request.form.get(f"commits{i}")),
                parse_float(request.form.get(f"commits_per_day{i}")),
                parse_float(request.form.get(f"forks_per_day{i}")),
                parse_float(request.form.get(f"days_since_update{i}")),
                parse_float(request.form.get(f"age_days{i}")),
                parse_float(request.form.get(f"has_homepage{i}")),
                parse_float(request.form.get(f"recently_updated{i}")),
            ]
            features.append(row)
            repos.append({"name": name})

        X = np.array(features)
        print("before",X)

        # === Call Celery task for predictions ===
        result = get_predictions(X)
        print("re",result) 

        # Attach predictions to corresponding repos
        for i in range(5):
            repos[i]["predicted_stars"] = int(result[i])

        # Sort repositories by predicted stars in descending order
        repos_sorted = sorted(repos, key=lambda x: x["predicted_stars"], reverse=True)

        return render_template('result.html', results=repos_sorted)

    except Exception as e:
        # Return error if any exception occurs
        return f"<h2>Error:</h2><p>{e}</p>"

if __name__ == '__main__':
    # Run Flask app
