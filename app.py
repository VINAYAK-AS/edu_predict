from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import json
import os
import warnings # <-- ADDED THIS
from dotenv import load_dotenv

# NEW IMPORTS FOR THE UPDATED SDK
from google import genai
from google.genai import types
from pydantic import BaseModel

# <-- ADDED THIS TO MUTE THE RED WARNINGS -->
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# 1. Load the hidden key from the .env file
load_dotenv()
my_api_key = os.getenv("GEMINI_API_KEY")

# 2. Initialize the new Gemini Client
client = genai.Client(api_key=my_api_key)

# 3. Define the exact JSON structure we want the AI to return
class CollegeDetails(BaseModel):
    courses: str
    fees: str
    reviews: str
    comments: str

# Our custom Cache Dictionary to memorize AI answers
college_cache = {}

# Load the model, encoders, and the dataset to get the names
try:
    model = joblib.load('models/keam_model.pkl')
    encoders = joblib.load('models/encoders.pkl')
    # Load data just to get unique college codes and names
    df = pd.read_csv('data/keam_master_dataset.csv')
    unique_colleges = df[['college_code', 'college_name']].drop_duplicates()
except Exception as e:
    print(f"Startup Error: {e}")

@app.route("/")
def home(): 
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sname = request.form.get("sname")
    user_rank = int(request.form.get("rank"))
    category = request.form.get("category")
    course = request.form.get("course")
    location_choice = request.form.get("location").lower()

    # 1. Dictionary
    district_keywords = {
        'eranakulam': ['ernakulam', 'kochi', 'cochin', 'thrikkakkara', 'kalamassery', 'aluva', 'angamaly', 'kothamangalam', 'muvattupuzha', 'perumbavoor', 'piravam', 'vazhakulam', 'kakkanad'],
        'trivandrum': ['trivandrum', 'thiruvananthapuram', 'tvm', 'tvpm', 'kazhakuttom', 'nedumangad', 'attingal', 'pappanamcode', 'barton hill', 'poojapura', 'vellanad', 'neyyattinkara', 'kallambalam'],
        'kottayam': ['kottayam', 'palai', 'kanjirapally', 'ettumanoor', 'pampady', 'changanassery', 'kidangoor', 'mattakara', 'puthuppally'],
        'trissur': ['thrissur', 'trissur', 'cheruthuruthy', 'irinjalakuda', 'chalakudy', 'mala', 'kodakara', 'wadakancherry', 'vallivattom'],
        'kollam': ['kollam', 'karunagappally', 'kottarakkara', 'punalur', 'chathannoor', 'sasthamcotta', 'parippally', 'ezhukone'],
        'palakkad': ['palakkad', 'shoranur', 'ottapalam', 'pattambi', 'lakkidi', 'sreekrishnapuram', 'pudussery'],
        'alapuzha': ['alappuzha', 'alapuzha', 'cherthala', 'chengannur', 'kuttanad', 'mavelikkara', 'punnapra', 'nooranadu'],
        'pathanamthitta': ['pathanamthitta', 'adoor', 'thiruvalla', 'kallooppara', 'aranmula'],
        'kozhikode': ['kozhikode', 'kozhikkode', 'calicut', 'mukkam', 'mukkom', 'vadakara', 'koyilandy', 'kakkodi'],
        'kannur': ['kannur', 'thalassery', 'payyannur', 'payyanur', 'pariyaram', 'chemperi'],
        'malappuaram': ['malappuram', 'kuttippuram', 'perinthalmanna', 'valanchery', 'manjeri', 'ponnani', 'tenhipalam'],
        'kasargod': ['kasaragod', 'kasargod', 'trikaripur', 'thrikarippur'],
        'wayanad': ['wayanad', 'mananthavady', 'pookode'],
        'idukki': ['idukki', 'thodupuzha', 'munnar', 'peermede']
    }

    search_terms = district_keywords.get(location_choice, [location_choice])

    # --- THE FIX: Get ONLY colleges that ACTUALLY offer the selected course ---
    valid_colleges_for_course = df[df['course'] == course]['college_code'].unique()
    target_colleges = set() 
    
    # 2. The For Loop: Check course FIRST, then location
    for index, row in unique_colleges.iterrows():
        college_code = row['college_code']
        college_name = str(row['college_name']).lower()
        
        # Step A: Does this college teach this course? If not, skip it!
        if college_code not in valid_colleges_for_course:
            continue
            
        # Step B: Is it in the right location?
        if any(term in college_name for term in search_terms):
            target_colleges.add(college_code)

    results_list = []

    # 3. Loop through the filtered colleges and predict
    for college_code in target_colleges:
        try:
            enc_college = encoders['college_code'].transform([college_code])[0]
            enc_course = encoders['course'].transform([course])[0]
            enc_cat = encoders['category'].transform([category])[0]
        
            input_data = np.array([[enc_college, enc_course, enc_cat, 2026, 3]])
            predicted_cutoff = model.predict(input_data)[0]
            
            rank_difference = predicted_cutoff - user_rank

            if rank_difference >= 5000:
                base_confidence = 85
            elif rank_difference >= 2500:
                base_confidence = 65
            elif rank_difference >= 500:
                base_confidence = 60
            elif rank_difference >= 0:
                base_confidence = 50
            elif rank_difference >= -1000:
                base_confidence = 45
            elif rank_difference >= -3000:
                base_confidence = 25
            else:
                base_confidence = 10

            raw_confidence = base_confidence + (rank_difference / 1000)
            confidence_pct = min(99, max(5, int(raw_confidence)))
            
            if confidence_pct >= 90:
                chance_label = "High Chance"
            elif confidence_pct >= 50:
                chance_label = "Medium Chance"
            else:
                chance_label = "Low Chance"

            # Grab the name for the HTML page
            full_name = unique_colleges[unique_colleges['college_code'] == college_code]['college_name'].values[0]

            results_list.append({
                'college_code': college_code,
                'college_name': full_name,
                'predicted_rank': int(predicted_cutoff),
                'confidence': confidence_pct,
                'chance': chance_label
            })
            
        except Exception as e:
            continue

    # Sort results
    results_list = sorted(results_list, key=lambda x: x['predicted_rank'], reverse=True)

    return render_template("result.html", 
                           name=sname, 
                           rank=user_rank, 
                           course=course,
                           location=location_choice.capitalize(),
                           results=results_list)

@app.route('/api/college-details/<college_code>')
def get_college_details(college_code):
    # 1. SPEED BOOST: Check if we already memorized this college!
    if college_code in college_cache:
        print(f"Serving {college_code} from cache!")
        return college_cache[college_code]

    # 2. Look up the full college name from your CSV data
    college_row = df[df['college_code'] == college_code]
    if college_row.empty:
        return {"courses": "Unknown", "fees": "Unknown", "reviews": "College not found in database.", "comments": ""}
        
    full_name = college_row['college_name'].values[0]

    # 3. Cleaned up prompt
    prompt = f"""
    Find the existing courses, approximate fee structure, and a 4-sentence general student review summary for {full_name} in Kerala. 
    Also add specific, individual student comments. Include a balanced mix of positive and negative feedback with student names.
    
    STRICT FORMATTING RULES:
    1. COURSES: Format as a vertical list using the bullet symbol (•). Put each course on a new line.
    2. FEES: Just state the exact rupee amounts clearly. (Do NOT use markdown asterisks like **).
    3. COMMENTS: Start every positive comment with a green circle (🟢) and every negative comment with a red circle (🔴).
    4. LINE BREAKS: You MUST separate every single comment with a double newline character (\\n\\n). Do NOT merge them into one line.
    """

    # 4. Ask the AI using the new Structured Outputs system
    try:
        print(f"Asking AI for {full_name} details...")
        
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CollegeDetails,
                temperature=0.3,
            ),
        )
        
        # Convert the AI's perfect JSON string into a Python dictionary
        ai_data = json.loads(response.text)
        
        # Save it to our cache memory
        college_cache[college_code] = ai_data
        
        return ai_data

    except Exception as e:
        print(f"AI API Error: {e}")
        # Graceful fallback
        return {
            "courses": "Unable to fetch from AI at the moment.", 
            "fees": "Unable to fetch from AI.", 
            "reviews": "The AI is currently resting. Please try again!",
            "comments": "No comments available."
        }

if __name__ == "__main__":
    app.run(debug=True)