from flask import Flask, request, render_template
import joblib
import numpy as np

# load the model
model = joblib.load('final_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        repos = []

        for i in range(1, 6):
            name = request.form.get(f'name{i}', f'Repo {i}')
            def parse_float(value):
                try:
                    return float(value)
                except:
                    return 0.0

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
        predictions = model.predict(X)

        for i in range(5):
            repos[i]["predicted_stars"] = int(predictions[i])

        # sort
        repos_sorted = sorted(repos, key=lambda x: x["predicted_stars"], reverse=True)

        return render_template('result.html', results=repos_sorted)

    except Exception as e:
        return f"<h2>Error:</h2><p>{e}</p>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)