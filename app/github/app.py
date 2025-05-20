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
            issues = float(request.form.get(f'issues{i}', 0))
            size = float(request.form.get(f'size{i}', 0))
            topics = float(request.form.get(f'topics{i}', 0))
            created_at = float(request.form.get(f'created_at{i}', 0))
            updated_at = float(request.form.get(f'updated_at{i}', 0))

            features.append([issues, size, topics, created_at, updated_at])
            repos.append({"name": name})

        X = np.array(features)
        predictions = model.predict(X)

        for i in range(5):
            repos[i]["predicted_stars"] = int(predictions[i])

        # 排序：按 predicted_stars 降序排列
        repos_sorted = sorted(repos, key=lambda x: x["predicted_stars"], reverse=True)

        return render_template('result.html', results=repos_sorted)

    except Exception as e:
        return f"<h2>Error:</h2><p>{e}</p>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)
