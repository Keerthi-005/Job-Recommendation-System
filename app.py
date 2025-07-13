from flask import Flask, jsonify, render_template, request, send_from_directory
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder='static')

# Global variables for ML model and data
model = None
job_data = []
job_texts = []

# Define the static folder path
STATIC_FOLDER = os.path.join(app.root_path, 'static')

def load_model():
    """Load the sentence transformer model for job recommendations"""
    global model
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence Transformer model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

def load_recommendation_jobs():
    """Load jobs data for ML-based recommendations"""
    global job_data, job_texts
    try:
        # Try to load from data/cleaned.json first
        data_path = 'data/cleaned.json'
        if not os.path.exists(data_path):
            # Fallback to static folder
            data_path = os.path.join(STATIC_FOLDER, 'cleaned.json')
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        job_data.clear()
        job_texts.clear()
        
        for entry in data:
            if not entry.get('Title1'):
                continue
            title = entry['Title1'].strip()
            job_text = f"Job Title: {title}"
            job_texts.append(job_text)
            job_data.append({
                'title': title
            })
        
        print(f"✅ Loaded {len(job_data)} jobs for recommendations.")
    except Exception as e:
        print(f"❌ Error loading recommendation jobs: {e}")
        job_data = []
        job_texts = []

@app.route('/')
def home():
    """Main page with unified interface"""
    return render_template('index.html')

@app.route('/bot')
def bot_page():
    """Career bot page"""
    return render_template('bot.html')

@app.route('/recommend')
def recommend_page():
    """Job recommendation page"""
    return render_template('recommend.html')

# Career Bot API Endpoints
@app.route('/api/categories')
def get_categories():
    """Get job categories for the career bot"""
    try:
        file_path = os.path.join(STATIC_FOLDER, 'bot.json')
        print(f"Loading categories from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            jobs_data = json.load(f)
        print(f"Jobs data loaded successfully. Keys: {list(jobs_data.keys())}")
        return jsonify(list(jobs_data.keys()))
    except Exception as e:
        print(f"Error loading categories: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/jobs')
def get_jobs_by_category():
    """Get jobs by category for the career bot"""
    category = request.args.get('category')
    try:
        file_path = os.path.join(STATIC_FOLDER, 'bot.json')
        print(f"Loading jobs from: {file_path} for category: {category}")
        with open(file_path, 'r', encoding='utf-8') as f:
            jobs_data = json.load(f)
        if category not in jobs_data:
            print(f"No jobs found for category: {category}")
            return jsonify([])
        print(f"Jobs for category {category}: {len(jobs_data[category])} jobs found")
        return jsonify(jobs_data[category])
    except Exception as e:
        print(f"Error loading jobs: {e}")
        return jsonify({"error": str(e)}), 500

# Job Recommendation API Endpoints
@app.route('/api/recommend-jobs', methods=['POST'])
def recommend_jobs():
    """AI-powered job recommendations based on user input"""
    try:
        data = request.json
        user_input = data.get('userInput', '') if data else ''

        if not user_input:
            return jsonify({"error": "Missing user input"}), 400
        
        if model is None:
            return jsonify({"error": "Recommendation model not available"}), 503
        
        if not job_texts:
            return jsonify({"error": "No job data available for recommendations"}), 503

        # Generate embeddings and calculate similarity scores
        input_embedding = model.encode(user_input)
        scores = []

        for i, job_text in enumerate(job_texts):
            try:
                job_embedding = model.encode(job_text)
                score = cosine_similarity([input_embedding], [job_embedding])[0][0]
                scores.append({"index": i, "score": float(score)})
            except Exception as e:
                print(f"Error processing job {i}: {e}")
                continue

        # Sort by score and get top recommendations
        scores.sort(key=lambda x: x['score'], reverse=True)
        top_jobs = scores[:5]  # Get top 5 instead of 3 for better variety

        recommendations = [
            {
                "title": job_data[item["index"]]["title"],
                "score": item["score"]
            }
            for item in top_jobs
        ]

        return jsonify(recommendations)
    
    except Exception as e:
        print(f"Error in recommend_jobs: {e}")
        return jsonify({"error": "Internal server error during recommendation"}), 500

# Health check endpoint
@app.route('/api/health')
def health_check():
    """Health check endpoint to verify system status"""
    status = {
        "status": "ok",
        "model_loaded": model is not None,
        "recommendation_jobs_count": len(job_data),
        "bot_data_available": os.path.exists(os.path.join(STATIC_FOLDER, 'bot.json'))
    }
    return jsonify(status)

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting Unified Career Assistant...")
    
    # Load ML model and data for recommendations
    load_model()
    load_recommendation_jobs()
    
    # Check if bot data exists
    bot_data_path = os.path.join(STATIC_FOLDER, 'bot.json')
    if os.path.exists(bot_data_path):
        print("✅ Career bot data found.")
    else:
        print("⚠️  Career bot data (bot.json) not found in static folder.")
    
    print("✅ Unified Career Assistant ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)
