"""
Flask Web Application for Skin Condition Classification
Supports single and multiple image uploads with predictions
"""

import os
import sys
import base64
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your predictor
try:
    from predict_skin_condition import SkinConditionPredictor, load_predictor
except ImportError:
    # Try relative import
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from predict_skin_condition import SkinConditionPredictor, load_predictor

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global predictor instance
predictor = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def init_predictor():
    """Initialize the predictor"""
    global predictor
    try:
        print("📦 Loading skin condition predictor...")
        predictor = load_predictor(model_type='auto', threshold=0.5, ensemble_method='weighted')
        print("✅ Predictor loaded successfully!")
        print(f"   Classes: {predictor.class_names}")
        return True
    except Exception as e:
        print(f"❌ Failed to load predictor: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_prediction_summary(result):
    """Extract summary from prediction result"""
    return {
        'predicted_class': str(result['predicted_class']),
        'confidence': f"{result['confidence']:.2%}",
        'confidence_value': float(result['confidence']),
        'is_confident': bool(result['is_confident']),
        'risk_level': 'High Risk' if result['confidence'] > 0.7 else 'Medium Risk' if result['confidence'] > 0.5 else 'Low Risk',
        'top_predictions': [
            {
                'class': str(p['class']),
                'probability': float(p['probability']),
                'confidence_percentage': p['confidence_percentage']
            }
            for p in result['top_predictions'][:3]
        ]
    }

def create_result_chart(result):
    """Create a bar chart of predictions"""
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get top 5 predictions
        top_preds = result['top_predictions'][:5]
        classes = [str(p['class']) for p in top_preds]
        probs = [float(p['probability']) for p in top_preds]
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(probs))]
        
        # Create horizontal bar chart
        bars = ax.barh(classes, probs, color=colors)
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title('Top Predictions', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for bar, prob in zip(bars, probs):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1%}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', 
                         classes=predictor.class_names if predictor else [],
                         models_available=predictor is not None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    try:
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        
        # Filter out empty files
        files = [f for f in files if f and f.filename]
        
        if not files:
            return jsonify({'error': 'No valid files selected'}), 400
        
        # Get threshold from request
        threshold = float(request.form.get('threshold', 0.5))
        predictor.threshold = threshold
        
        # Get model type
        model_type = request.form.get('model_type', 'auto')
        
        # Process each file
        results = []
        for file in files:
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': f'File type not allowed. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
                })
                continue
            
            try:
                # Save file temporarily
                filename = secure_filename(file.filename)
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
                file.save(temp_path)
                
                print(f"Processing: {filename}")
                
                # Make prediction
                result = predictor.predict(temp_path, return_details=True)
                
                # Convert numpy types to Python native types
                result = convert_to_serializable(result)
                
                # Get prediction summary
                summary = get_prediction_summary(result)
                
                # Create visualization chart
                chart_base64 = create_result_chart(result)
                
                # Convert image to base64 for display
                with open(temp_path, 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                
                results.append({
                    'filename': filename,
                    'success': True,
                    'prediction': summary,
                    'image_base64': img_base64,
                    'chart_base64': chart_base64,
                    'detailed_results': {
                        'all_probabilities': convert_to_serializable(result.get('all_probabilities', {})),
                        'model_used': str(result.get('model_used', 'unknown')),
                        'ensemble_method': str(result.get('ensemble_method')) if result.get('ensemble_method') else None,
                        'models_used': [str(m) for m in result.get('models_used', [])]
                    }
                })
                
                # Clean up temp file
                os.remove(temp_path)
                
            except Exception as e:
                print(f"Error processing {file.filename}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total_images': len(results),
            'results': results,
            'class_names': predictor.class_names
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Handle batch prediction from folder (zip file)"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if it's a zip file
        if not file.filename.endswith('.zip'):
            return jsonify({'error': 'Please upload a ZIP file'}), 400
        
        import zipfile
        
        # Save zip file
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.zip")
        file.save(zip_path)
        
        # Extract zip
        extract_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Process all images in the extracted folder
        results = []
        for root, dirs, files in os.walk(extract_path):
            for filename in files:
                if allowed_file(filename):
                    try:
                        img_path = os.path.join(root, filename)
                        result = predictor.predict(img_path, return_details=True)
                        summary = get_prediction_summary(result)
                        
                        results.append({
                            'filename': filename,
                            'success': True,
                            'prediction': summary['predicted_class'],
                            'confidence': summary['confidence'],
                            'is_confident': bool(summary['is_confident'])
                        })
                    except Exception as e:
                        results.append({
                            'filename': filename,
                            'success': False,
                            'error': str(e)
                        })
        
        # Clean up
        import shutil
        os.remove(zip_path)
        shutil.rmtree(extract_path)
        
        # Generate summary statistics
        successful = [r for r in results if r.get('success', False)]
        if successful:
            df = pd.DataFrame(successful)
            summary_stats = {
                'total': len(results),
                'successful': len(successful),
                'confident_predictions': len([r for r in successful if r.get('is_confident', False)]),
                'class_distribution': convert_to_serializable(df['prediction'].value_counts().to_dict()) if 'prediction' in df.columns else {}
            }
        else:
            summary_stats = {'total': len(results), 'successful': 0}
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': summary_stats
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'classes': predictor.class_names if predictor else [],
        'num_classes': len(predictor.class_names) if predictor else 0
    })

@app.route('/class_info')
def class_info():
    """Get class information"""
    if predictor:
        return jsonify({
            'classes': predictor.class_names,
            'num_classes': len(predictor.class_names)
        })
    return jsonify({'error': 'Model not loaded'}), 500

# Initialize predictor when app starts
print("Starting Skin Condition Classifier Web App...")
init_success = init_predictor()
if not init_success:
    print("⚠️ WARNING: Predictor failed to load. Please check your model files.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)