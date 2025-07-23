from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import requests
import pandas as pd
import ibm_boto3
from botocore.client import Config
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Watsonx credentials
API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
MODEL_ID = os.getenv("MODEL_ID")

# COS credentials
COS_API_KEY = os.getenv("COS_API_KEY")
COS_RESOURCE_INSTANCE_ID = os.getenv("COS_RESOURCE_INSTANCE_ID")
COS_ENDPOINT = os.getenv("COS_ENDPOINT")
COS_BUCKET = os.getenv("COS_BUCKET")
COS_FILE = os.getenv("COS_FILE")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Debug: Confirm environment variables are loading
print("COS_BUCKET:", COS_BUCKET)
print("COS_FILE:", COS_FILE)
print("AWS_ACCESS_KEY_ID:", AWS_ACCESS_KEY_ID)

# Setup COS client
cos_client = ibm_boto3.client("s3",
    ibm_api_key_id=COS_API_KEY,
    ibm_service_instance_id=COS_RESOURCE_INSTANCE_ID,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Get IAM access token
def get_access_token():
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={API_KEY}&grant_type=urn:ibm:params:oauth:grant-type:apikey"
    response = requests.post(url, headers=headers, data=data)
    return response.json()["access_token"]

# Generate meal plan using Watsonx REST API
def generate_meal_plan(prompt):
    access_token = get_access_token()
    url = f"https://{REGION}.ml.cloud.ibm.com/ml/v1/text/generation?version=2024-05-28"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    body = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "input": prompt
    }
    response = requests.post(url, headers=headers, json=body)
    return response.json()["results"][0]["generated_text"]

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        age = data.get("age", 25)
        diet = data.get("diet", "vegetarian")
        goal = data.get("goal", "weight loss")
        allergies = data.get("allergies", [])

        # Load file from COS
        response = cos_client.get_object(Bucket=COS_BUCKET, Key=COS_FILE)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))

        # Build prompt
        prompt = f"""
        I am a {age}-year-old person on a {diet} diet.
        I am allergic to {', '.join(allergies) if allergies else 'none'}.
        My goal is {goal}.
        Generate a detailed 1500-calorie meal plan for breakfast, lunch, and dinner with nutritional explanation.
        """

        meal_plan = generate_meal_plan(prompt)

        return jsonify({
            "input": data,
            "meal_plan": meal_plan
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
