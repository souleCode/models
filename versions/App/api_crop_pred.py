from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
import os


app = Flask(__name__)

# Configuration des parametres pour le deploiment
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')
app.config['PORT'] = os.environ.get('PORT_API_PRED')

# Tente de charger le modèle
try:
    # Utilise le nom du fichier pour le charger
    # Remplace f par "model_pipeline.pkl"
    loaded_model = joblib.load('../Models/model_pipeline_v1.pkl')

    @app.route('/predict', methods=['POST'])
    def predict():
        json_ = request.json
        query_df = pd.DataFrame(json_)

        # Obtenir les prédictions
        y_reg_pred = loaded_model.predict(query_df)

        # Convertir les prédictions en listes
        return jsonify({
            'prediction': y_reg_pred[0]
        })

except Exception as e:
    print("Error loading model:", e)


if __name__ == "__main__":
    app.run(port=5001)
