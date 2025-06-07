import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Email Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
        width: 100%;
    }
    
    .logo-container img {
        max-width: 400px;
        width: 100%;
        height: auto;
        object-fit: contain;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .input-section {
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .result-success {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .result-danger {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* Primary button styling - multiple selectors for compatibility */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"],
    .stButton button[kind="primary"],
    .stButton button {
        background-color: #28a745 !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        font-weight: bold !important;
        width: 100% !important;
        box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3) !important;
        transition: all 0.3s ease !important;
        font-size: 1rem !important;
    }
    
    /* Hover effects */
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover,
    .stButton button[kind="primary"]:hover,
    .stButton button:hover {
        background-color: #218838 !important;
        box-shadow: 0 6px 12px rgba(40, 167, 69, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Additional button styling for Streamlit's newer versions */
    div[data-testid="stButton"] > button {
        background-color: #28a745 !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        font-weight: bold !important;
        width: 100% !important;
        box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stButton"] > button:hover {
        background-color: #218838 !important;
        box-shadow: 0 6px 12px rgba(40, 167, 69, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    .model-results {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .model-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .model-card {
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid transparent;
    }
    
    .model-svm {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .model-rf {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .model-nb {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    
    .model-lr {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    
    .model-mlp {
        background-color: #fce4ec;
        border-left: 4px solid #e91e63;
    }
    
    .ensemble-card {
        background-color: #f0f4f8;
        border: 2px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Additional responsive styling for mobile */
    @media (max-width: 768px) {
        .logo-container img {
            max-width: 300px;
        }
    }
    
    @media (max-width: 480px) {
        .logo-container img {
            max-width: 250px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        models = {}
        
        with open('svm_model.pkl', 'rb') as f:
            models['svm'] = pickle.load(f)
        
        with open('rf_model.pkl', 'rb') as f:
            models['rf'] = pickle.load(f)
        
        with open('naive_model.pkl', 'rb') as f:
            models['naive'] = pickle.load(f)
        
        with open('lr_model.pkl', 'rb') as f:
            models['lr'] = pickle.load(f)
        
        with open('mlp_model.pkl', 'rb') as f:
            models['mlp'] = pickle.load(f)
        
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            models['tfidf'] = pickle.load(f)
        
        with open('onehot_encoder.pkl', 'rb') as f:
            models['encoder'] = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)
        
        with open('minmax_scaler.pkl', 'rb') as f:
            models['minmax_scaler'] = pickle.load(f)
        
        return models
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load models
models = load_models()

# Header with centered logo using columns for better control
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    try:
        st.image("phishshield_logo.png", width=350)
    except:
        st.error("Logo file 'phishshield_logo.png' not found. Please ensure the file is in the same directory as this script.")

st.markdown('<p class="sub-header">Analyze emails to detect potential phishing attempts using multiple machine learning models</p>', unsafe_allow_html=True)

# Information section
with st.expander("How to use this tool", expanded=False):
    st.markdown("""
    **Instructions:**
    1. **Sender Email**: Enter the complete email address of the sender
    2. **Day of Week**: Select the day when the email was received
    3. **Hour**: Enter the hour when the email was received (0-23 format)
    4. **Email Content**: Paste the entire email body content
    5. Click **Analyze Email** to get predictions from all four models
    
    **Models Used:**
    - **SVM (Support Vector Machine)**: Linear classifier optimized for text classification
    - **Random Forest**: Ensemble method using multiple decision trees
    - **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
    - **Logistic Regression**: Linear classifier using logistic function
    - **MLP (Multi-Layer Perceptron)**: Neural network with multiple hidden layers
    
    **Note**: All fields are required for accurate prediction.
    """)

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("Email Details")
    
    # Email input with validation
    input_sender_email = st.text_input(
        "Sender Email Address",
        placeholder="example@domain.com",
        help="Enter the complete email address of the sender"
    )
    
    # Get current day and hour
    current_time = datetime.now()
    current_day_index = current_time.weekday()  # Monday is 0, Sunday is 6
    current_hour = current_time.hour
    
    # Day of week with selectbox
    day_options = {
        "Monday": 0,
        "Tuesday": 1, 
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }
    
    # Get current day name as default
    day_names = list(day_options.keys())
    current_day_name = day_names[current_day_index]
    
    selected_day = st.selectbox(
        "Day of the Week",
        options=list(day_options.keys()),
        index=current_day_index,
        help="Select the day when the email was received (auto-set to today)"
    )
    input_day_of_week = day_options[selected_day]
    
    # Hour input with number input
    input_hour = st.number_input(
        "Hour (24-hour format)",
        min_value=0,
        max_value=23,
        value=current_hour,
        help="Enter the hour when the email was received (auto-set to current hour)"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("Email Content")
    
    # Email body input
    input_body = st.text_area(
        "Email Body",
        placeholder="Paste the complete email content here...",
        height=200,
        help="Enter the complete email body content"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Validation function
def validate_inputs(email, body):
    errors = []
    
    if not email.strip():
        errors.append("Email address is required")
    elif '@' not in email:
        errors.append("Invalid email format - missing @ symbol")
    elif len(email.split('@')) != 2:
        errors.append("Invalid email format")
    elif not email.split('@')[0] or not email.split('@')[1]:
        errors.append("Invalid email format - empty username or domain")
    
    if not body.strip():
        errors.append("Email body is required")
    
    return errors

# Prediction function with all five models
def predict_phishing_all_models(sender_email, day_of_week, email_body, hour):
    try:
        # Split email
        email_parts = sender_email.split('@')
        if len(email_parts) != 2:
            raise ValueError("Invalid email format")
        
        email_name, email_domain = email_parts
        
        # URL detection
        url_pattern = r'https?://\S+'
        urls = re.findall(url_pattern, email_body)
        has_url = 1 if urls else 0
        
        # Feature extraction
        x_text = models['tfidf'].transform([email_body])
        x_cat = models['encoder'].transform([[email_name, email_domain]])
        x_num = models['scaler'].transform([[has_url, hour, day_of_week]])
        x_num2 = models['minmax_scaler'].transform([[has_url, hour, day_of_week]])
        
        # Make predictions with all models
        results = {}
        model_names = {
            'svm': 'SVM',
            'rf': 'Random Forest',
            'naive': 'Naive Bayes',
            'lr': 'Logistic Regression',
            # 'mlp': 'MLP Neural Network'
            'mlp': 'MLP'
        }
        
        predictions = []
        
        for model_key, model_name in model_names.items():
            # Use x_num2 (minmax scaled) only for naive model, x_num for others
            if model_key == 'naive':
                x_final = hstack([x_text, x_cat, x_num2])
            else:
                x_final = hstack([x_text, x_cat, x_num])
            
            prediction = models[model_key].predict(x_final)[0]
            probability = None
            confidence = None
            
            if hasattr(models[model_key], 'predict_proba'):
                probability = models[model_key].predict_proba(x_final)[0]
                confidence = max(probability) * 100
            
            results[model_key] = {
                'name': model_name,
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence
            }
            predictions.append(prediction)
        
        # Ensemble prediction (majority vote)
        phishing_votes = sum(predictions)
        ensemble_prediction = 1 if phishing_votes >= 3 else 0
        
        return results, ensemble_prediction, has_url, len(urls), phishing_votes
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None, None

# Analysis section
st.markdown("---")
st.subheader("Analysis")

# Create columns for the button and results
button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:
    analyze_button = st.button("Analyze Email", type="primary")

if analyze_button:
    # Validate inputs
    validation_errors = validate_inputs(input_sender_email, input_body)
    
    if validation_errors:
        st.error("Please fix the following errors:")
        for error in validation_errors:
            st.error(f"• {error}")
    else:
        # Show processing message
        with st.spinner("Analyzing email with all five machine learning models..."):
            model_results, ensemble_prediction, has_url, url_count, phishing_votes = predict_phishing_all_models(
                input_sender_email, 
                input_day_of_week, 
                input_body, 
                input_hour
            )
        
        if model_results is not None:
            # Ensemble Result - Removed the white box by directly showing the result
            if ensemble_prediction == 1:
                st.markdown("""
                <div class="result-danger">
                    <h3>🚨 PHISHING DETECTED</h3>
                    <p>This email appears to be a phishing attempt. Exercise caution!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-success">
                    <h3>✅ LEGITIMATE EMAIL</h3>
                    <p>This email appears to be legitimate.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Individual Model Results in expander
            with st.expander("Individual Model Results", expanded=False):
                st.subheader("Detailed Model Analysis")
                
                # Create grid for models (3 rows, 2 columns for 5 models)
                model_styles = {
                    'svm': 'model-svm',
                    'rf': 'model-rf',
                    'naive': 'model-nb',
                    'lr': 'model-lr',
                    'mlp': 'model-mlp'
                }
                
                model_keys = list(model_results.keys())
                
                # First row
                col1, col2 = st.columns(2)
                
                with col1:
                    model_key = model_keys[0]  # SVM
                    st.markdown(f'<div class="model-card {model_styles[model_key]}">', unsafe_allow_html=True)
                    st.markdown(f"**{model_results[model_key]['name']}**")
                    if model_results[model_key]['prediction'] == 1:
                        result = "Phishing ❌"
                    else:
                        result = "Legitimate ✅"
                    st.write(f"**Prediction**: {result}")
                    if model_results[model_key]['confidence']:
                        st.write(f"**Confidence**: {model_results[model_key]['confidence']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    model_key = model_keys[1]  # Random Forest
                    st.markdown(f'<div class="model-card {model_styles[model_key]}">', unsafe_allow_html=True)
                    st.markdown(f"**{model_results[model_key]['name']}**")
                    if model_results[model_key]['prediction'] == 1:
                        result = "Phishing ❌"
                    else:
                        result = "Legitimate ✅"
                    st.write(f"**Prediction**: {result}")
                    if model_results[model_key]['confidence']:
                        st.write(f"**Confidence**: {model_results[model_key]['confidence']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Second row
                col3, col4 = st.columns(2)
                
                with col3:
                    model_key = model_keys[2]  # Naive Bayes
                    st.markdown(f'<div class="model-card {model_styles[model_key]}">', unsafe_allow_html=True)
                    st.markdown(f"**{model_results[model_key]['name']}**")
                    if model_results[model_key]['prediction'] == 1:
                        result = "Phishing ❌"
                    else:
                        result = "Legitimate ✅"
                    st.write(f"**Prediction**: {result}")
                    if model_results[model_key]['confidence']:
                        st.write(f"**Confidence**: {model_results[model_key]['confidence']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    model_key = model_keys[3]  # Logistic Regression
                    st.markdown(f'<div class="model-card {model_styles[model_key]}">', unsafe_allow_html=True)
                    st.markdown(f"**{model_results[model_key]['name']}**")
                    if model_results[model_key]['prediction'] == 1:
                        result = "Phishing ❌"
                    else:
                        result = "Legitimate ✅"
                    st.write(f"**Prediction**: {result}")
                    if model_results[model_key]['confidence']:
                        st.write(f"**Confidence**: {model_results[model_key]['confidence']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Third row (centered)
                col5, col6, col7 = st.columns([1, 1, 1])
                
                with col6:
                    model_key = model_keys[4]  # MLP
                    st.markdown(f'<div class="model-card {model_styles[model_key]}">', unsafe_allow_html=True)
                    st.markdown(f"**{model_results[model_key]['name']}**")
                    if model_results[model_key]['prediction'] == 1:
                        result = "Phishing ❌"
                    else:
                        result = "Legitimate ✅"
                    st.write(f"**Prediction**: {result}")
                    if model_results[model_key]['confidence']:
                        st.write(f"**Confidence**: {model_results[model_key]['confidence']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Model Agreement Analysis
                st.subheader("Model Agreement Analysis")
                agreement_col1, agreement_col2 = st.columns([1, 1])
                
                with agreement_col1:
                    st.write(f"**Votes for Phishing**: {phishing_votes}/5")
                    st.write(f"**Votes for Legitimate**: {5 - phishing_votes}/5")
                
                with agreement_col2:
                    if phishing_votes == 5 or phishing_votes == 0:
                        st.success("**Unanimous Agreement**: All models agree")
                    elif phishing_votes == 4 or phishing_votes == 1:
                        st.info("**Strong Consensus**: 4 out of 5 models agree")
                    elif phishing_votes == 3 or phishing_votes == 2:
                        st.warning("**Mixed Results**: Models are divided")
            
            # Analysis Details (always shown)
            st.subheader("Analysis Details")
            details_col1, details_col2 = st.columns([1, 1])
            
            with details_col1:
                st.markdown("**Email Information:**")
                st.write(f"• **Sender Domain**: {input_sender_email.split('@')[1]}")
                st.write(f"• **Day**: {selected_day}")
                st.write(f"• **Hour**: {input_hour}:00")
            
            with details_col2:
                st.markdown("**Content Analysis:**")
                st.write(f"• **URLs Found**: {url_count}")
                st.write(f"• **Contains URLs**: {'Yes' if has_url else 'No'}")
                st.write(f"• **Email Length**: {len(input_body)} characters")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Copyright PhishShield team</p>
    <p>🔬 Using ensemble of SVM, Random Forest, Naive Bayes, Logistic Regression, and MLP models for improved accuracy</p>
</div>
""", unsafe_allow_html=True)