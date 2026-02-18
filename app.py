from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

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

    # 1. Dictionary (Removed "pala" to stop it from matching "palakkad"!)
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
    # Using a 'set' automatically prevents duplicate colleges from being added!
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
            raw_confidence = 75 + (rank_difference / 100)
            confidence_pct = min(99, max(5, int(raw_confidence)))

            if confidence_pct >= 80:
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
    results_list = sorted(results_list, key=lambda x: x['confidence'], reverse=True)

    return render_template("result.html", 
                           name=sname, 
                           rank=user_rank, 
                           course=course,
                           location=location_choice.capitalize(),
                           results=results_list)
if __name__ == "__main__":
    app.run(debug=True)