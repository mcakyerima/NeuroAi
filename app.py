from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import os
import time
import numpy as np
from robust_brain_tumor_predictor import RobustBrainTumorPredictor
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictor
try:
    predictor = RobustBrainTumorPredictor()
    print("✅ Robust Brain Tumor Predictor loaded successfully!")
except Exception as e:
    print(f"❌ Error loading predictor: {e}")
    predictor = None

# Original HTML template (keeping your smooth design)
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroAI - Advanced Brain Tumor Detection</title>
    
    <!-- PWA Meta Tags -->
    <meta name="theme-color" content="#0a0a0a">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="NeuroAI">
    
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        :root {
            --primary-bg: #0a0a0a;
            --secondary-bg: #1a1a1a;
            --accent-bg: #2a2a2a;
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            --neon-blue: #00f5ff;
            --neon-purple: #bf00ff;
            --neon-green: #00ff41;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-touch-callout: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            user-select: none;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--primary-bg);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            touch-action: manipulation;
        }

        /* Animated background */
        .bg-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--primary-bg);
            z-index: -2;
        }

        .bg-animation::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 20%, rgba(0, 245, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(191, 0, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(0, 255, 65, 0.05) 0%, transparent 50%);
            animation: backgroundPulse 8s ease-in-out infinite alternate;
        }

        @keyframes backgroundPulse {
            0% { opacity: 0.5; transform: scale(1); }
            100% { opacity: 1; transform: scale(1.1); }
        }

        .container {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
            gap: 20px;
        }

        .header {
            text-align: center;
            padding: 20px;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 25px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
            animation: headerShimmer 3s infinite;
        }

        @keyframes headerShimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }

        .logo {
            font-size: 3.5em;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
            text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
            animation: logoGlow 3s ease-in-out infinite alternate;
        }

        @keyframes logoGlow {
            0% { filter: brightness(1); }
            100% { filter: brightness(1.2); }
        }

        .subtitle {
            font-size: 1.2em;
            color: var(--text-secondary);
            font-weight: 300;
            letter-spacing: 1px;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            min-height: 500px;
        }

        .upload-section, .preview-section {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 25px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            padding: 30px;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .upload-section:hover, .preview-section:hover {
            transform: translateY(-5px);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.4);
        }

        .upload-section-content, .preview-section-content {
            display: flex;
            flex-direction: column;
            height: 100%;
            position: relative;
            z-index: 2;
        }

        .section-title {
            font-size: 1.4em;
            font-weight: 600;
            margin-bottom: 25px;
            color: var(--neon-blue);
            text-align: center;
            position: relative;
        }

        .section-title::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 2px;
            background: var(--primary-gradient);
            border-radius: 2px;
        }

        .upload-options {
            display: flex;
            flex-direction: column;
            gap: 25px;
            flex: 1;
        }

        .upload-area {
            border: 2px dashed var(--glass-border);
            border-radius: 20px;
            padding: 40px 20px;
            text-align: center;
            background: var(--glass-bg);
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            flex: 1;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            touch-action: manipulation;
        }

        .upload-area:hover, .upload-area.dragover {
            border-color: var(--neon-blue);
            background: rgba(0, 245, 255, 0.1);
            transform: scale(1.02);
            box-shadow: 0 0 30px rgba(0, 245, 255, 0.3);
        }

        .upload-area::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }

        .upload-icon {
            font-size: 4em;
            margin-bottom: 15px;
            color: var(--neon-blue);
            animation: iconFloat 3s ease-in-out infinite;
        }

        @keyframes iconFloat {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        .upload-text {
            font-size: 1.3em;
            font-weight: 500;
            margin-bottom: 8px;
            color: var(--text-primary);
        }

        .upload-subtext {
            color: var(--text-secondary);
            font-size: 0.9em;
        }

        #fileInput {
            display: none;
        }

        .camera-btn {
            background: linear-gradient(45deg, var(--neon-green), var(--neon-blue));
            color: var(--primary-bg);
            border: none;
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px rgba(0, 255, 65, 0.3);
            touch-action: manipulation;
        }

        .camera-btn:hover, .camera-btn:active {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(0, 255, 65, 0.4);
        }

        .image-preview {
            display: none;
            text-align: center;
            margin-bottom: 25px;
        }

        .preview-img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            margin-bottom: 20px;
            border: 2px solid var(--glass-border);
            transition: all 0.3s ease;
        }

        .preview-img:hover {
            transform: scale(1.02);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.6);
        }

        .analyze-btn {
            background: var(--primary-gradient);
            color: white;
            border: none;
            padding: 18px 50px;
            font-size: 1.2em;
            font-weight: 700;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
            text-transform: uppercase;
            letter-spacing: 1px;
            width: 100%;
            margin: 20px 0;
            touch-action: manipulation;
            position: relative;
            overflow: hidden;
        }

        .analyze-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }

        .analyze-btn:hover::before {
            left: 100%;
        }

        .analyze-btn:hover, .analyze-btn:active {
            transform: translateY(-3px);
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.6);
        }

        .analyze-btn:disabled {
            background: var(--accent-bg);
            color: var(--text-secondary);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 25px;
        }

        .loading-spinner {
            width: 100px;
            height: 100px;
            border: 4px solid var(--glass-border);
            border-top: 4px solid var(--neon-blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 25px;
            position: relative;
        }

        .loading-spinner::after {
            content: '';
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            bottom: 10px;
            border: 2px solid transparent;
            border-top: 2px solid var(--neon-purple);
            border-radius: 50%;
            animation: spin 2s linear infinite reverse;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            font-size: 1.2em;
            color: var(--neon-blue);
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }

        .results-container {
            display: none;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 25px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
            animation: slideInUp 0.6s ease-out;
            margin-top: 20px;
        }

        .results-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--primary-gradient);
            border-radius: 25px 25px 0 0;
        }

        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .results-header {
            text-align: center;
            margin-bottom: 35px;
        }

        .results-title {
            font-size: 2.2em;
            font-weight: 800;
            color: var(--neon-green);
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(0, 255, 65, 0.3);
        }

        .results-subtitle {
            font-size: 1em;
            color: var(--text-secondary);
            font-weight: 300;
        }

        .results-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 40px;
            align-items: start;
        }

        .prediction-summary {
            background: var(--accent-bg);
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            border: 1px solid var(--glass-border);
            position: relative;
            overflow: hidden;
        }

        .prediction-summary::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--primary-gradient);
        }

        .prediction-label {
            font-size: 0.9em;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
        }

        .prediction-value {
            font-size: 2.2em;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 2px;
            line-height: 1.2;
        }

        .confidence-value {
            font-size: 1.3em;
            color: var(--neon-blue);
            font-weight: 600;
        }

        .confidence-level {
            font-size: 0.9em;
            color: var(--text-secondary);
            margin-top: 5px;
        }

        .detailed-analysis {
            background: var(--accent-bg);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid var(--glass-border);
        }

        .analysis-title {
            font-size: 1.3em;
            color: var(--neon-purple);
            margin-bottom: 25px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .probability-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }

        .probability-card {
            background: var(--glass-bg);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid var(--glass-border);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .probability-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
        }

        .probability-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--primary-gradient);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .probability-card:hover::before {
            opacity: 1;
        }

        .probability-name {
            font-size: 0.9em;
            color: var(--text-primary);
            margin-bottom: 15px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .circular-progress {
            position: relative;
            width: 80px;
            height: 80px;
            margin: 0 auto 15px;
        }

        .circular-progress svg {
            width: 100%;
            height: 100%;
            transform: rotate(-90deg);
        }

        .circular-progress circle {
            fill: none;
            stroke-width: 6;
            stroke-linecap: round;
        }

        .circular-progress .track {
            stroke: var(--glass-border);
        }

        .circular-progress .progress {
            stroke: url(#progressGradient);
            stroke-dasharray: 251;
            stroke-dashoffset: 251;
            transition: stroke-dashoffset 1.5s ease-out;
        }

        .circular-progress .percentage {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 16px;
            font-weight: 700;
            color: var(--text-primary);
            text-align: center;
        }

        .reset-section {
            text-align: center;
            margin-top: 30px;
            padding-top: 30px;
            border-top: 1px solid var(--glass-border);
        }

        .reset-btn {
            background: var(--accent-bg);
            color: var(--text-primary);
            border: 1px solid var(--glass-border);
            padding: 15px 40px;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            font-size: 1em;
            text-transform: uppercase;
            letter-spacing: 1px;
            touch-action: manipulation;
            position: relative;
            overflow: hidden;
        }

        .reset-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.5s;
        }

        .reset-btn:hover::before {
            left: 100%;
        }

        .reset-btn:hover, .reset-btn:active {
            background: var(--glass-bg);
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        }

        .error {
            display: none;
            background: linear-gradient(45deg, #ff6b6b, #ee5a52);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            text-align: center;
            font-weight: 500;
            box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
        }

        .camera-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            backdrop-filter: blur(10px);
        }

        .camera-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 25px;
            padding: 30px;
            text-align: center;
            max-width: 90%;
            max-height: 90%;
        }

        #cameraVideo {
            width: 100%;
            max-width: 500px;
            border-radius: 15px;
            margin-bottom: 20px;
        }

        .camera-controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .capture-btn, .close-camera-btn {
            padding: 12px 25px;
            border: none;
            border-radius: 25px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            touch-action: manipulation;
        }

        .capture-btn {
            background: var(--primary-gradient);
            color: white;
        }

        .close-camera-btn {
            background: var(--accent-bg);
            color: var(--text-primary);
        }

        /* Mobile Responsive */
        @media screen and (max-width: 768px) {
            .container {
                padding: 15px;
                gap: 15px;
            }

            .main-content {
                grid-template-columns: 1fr;
                gap: 20px;
                min-height: auto;
            }
            
            .logo {
                font-size: 2.8em;
            }
            
            .subtitle {
                font-size: 1em;
            }
            
            .upload-section, .preview-section {
                padding: 25px 20px;
            }
            
            .upload-area {
                padding: 30px 20px;
                min-height: 180px;
            }
            
            .upload-icon {
                font-size: 3.5em;
            }
            
            .upload-text {
                font-size: 1.2em;
            }
            
            .analyze-btn {
                padding: 18px 40px;
                font-size: 1.1em;
            }
            
            .camera-content {
                max-width: 95%;
                padding: 20px;
            }
            
            .camera-controls {
                flex-direction: column;
                align-items: center;
            }
            
            .capture-btn, .close-camera-btn {
                width: 200px;
            }

            .results-container {
                padding: 25px 20px;
            }

            .results-content {
                grid-template-columns: 1fr;
                gap: 25px;
            }

            .probability-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
            }

            .circular-progress {
                width: 70px;
                height: 70px;
            }

            .circular-progress .percentage {
                font-size: 14px;
            }

            .prediction-value {
                font-size: 1.8em;
            }

            .results-title {
                font-size: 1.8em;
            }
        }

        /* Touch-friendly hover states */
        @media (hover: none) and (pointer: coarse) {
            .upload-area:hover {
                transform: none;
                box-shadow: none;
            }
            
            .analyze-btn:hover, .camera-btn:hover, .reset-btn:hover {
                transform: none;
            }
        }
    </style>
</head>
<body>
    <div class="bg-animation"></div>
    
    <div class="container">
        <div class="header">
            <div class="logo">🧠 NeuroAI</div>
            <div class="subtitle">Advanced Neural Tumor Detection System</div>
        </div>

        <div class="main-content">
            <div class="upload-section">
                <div class="upload-section-content">
                    <h2 class="section-title">📤 Input Source</h2>
                    
                    <div class="upload-options">
                        <div class="upload-area" id="uploadArea">
                            <div class="upload-icon">🎯</div>
                            <div class="upload-text">Drag & Drop MRI Scan</div>
                            <div class="upload-subtext">or tap to browse files</div>
                            <input type="file" id="fileInput" accept="image/*">
                        </div>
                        
                        <button class="camera-btn" id="cameraBtn">
                            📷 Capture with Camera
                        </button>
                    </div>
                </div>
            </div>

            <div class="preview-section">
                <div class="preview-section-content">
                    <h2 class="section-title">🔍 Analysis</h2>
                    
                    <div class="image-preview" id="imagePreview">
                        <img id="previewImg" class="preview-img" alt="Preview">
                    </div>
                    
                    <div style="text-align: center; padding: 40px; color: var(--text-secondary);" id="emptyState">
                        <div style="font-size: 3em; margin-bottom: 15px;">🤖</div>
                        <div>Upload an MRI scan to begin AI analysis</div>
                    </div>
                    
                    <button class="analyze-btn" id="analyzeBtn" disabled>
                        🚀 Initialize Neural Analysis
                    </button>
                    
                    <div class="loading" id="loading">
                        <div class="loading-spinner"></div>
                        <div class="loading-text">AI Processing Neural Patterns...</div>
                    </div>
                    
                    <div class="error" id="error"></div>
                </div>
            </div>
        </div>

        <!-- Results Container -->
        <div class="results-container" id="resultsContainer">
            <div class="results-header">
                <div class="results-title">⚡ Neural Analysis Complete</div>
                <div class="results-subtitle">Advanced AI diagnosis and probability mapping</div>
            </div>

            <div class="results-content">
                <div class="prediction-summary">
                    <div class="prediction-label">Primary Diagnosis</div>
                    <div class="prediction-value" id="predictionValue"></div>
                    <div class="confidence-value" id="confidenceValue"></div>
                    <div class="confidence-level" id="confidenceLevel"></div>
                </div>

                <div class="detailed-analysis">
                    <div class="analysis-title">
                        🧬 Detailed Neural Mapping
                    </div>
                    <svg width="0" height="0">
                        <defs>
                            <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
                                <stop offset="50%" style="stop-color:#764ba2;stop-opacity:1" />
                                <stop offset="100%" style="stop-color:#f093fb;stop-opacity:1" />
                            </linearGradient>
                        </defs>
                    </svg>
                    <div class="probability-grid" id="probabilityGrid"></div>
                </div>
            </div>

            <div class="reset-section">
                <button class="reset-btn" onclick="resetApp()">
                    🔄 Analyze New Scan
                </button>
            </div>
        </div>
    </div>

    <!-- Camera Modal -->
    <div class="camera-modal" id="cameraModal">
        <div class="camera-content">
            <h3 style="margin-bottom: 20px; color: var(--neon-blue);">📷 Neural Scan Capture</h3>
            <video id="cameraVideo" autoplay playsinline></video>
            <canvas id="cameraCanvas" style="display: none;"></canvas>
            <div class="camera-controls">
                <button class="capture-btn" id="captureBtn">📸 Capture Scan</button>
                <button class="close-camera-btn" id="closeCameraBtn">❌ Close</button>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let selectedFile = null;
        let stream = null;
        const API_URL = window.location.origin + '/predict';
        
        // DOM elements
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const imagePreview = document.getElementById('imagePreview');
        const previewImg = document.getElementById('previewImg');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const loading = document.getElementById('loading');
        const resultsContainer = document.getElementById('resultsContainer');
        const error = document.getElementById('error');
        const emptyState = document.getElementById('emptyState');
        const cameraBtn = document.getElementById('cameraBtn');
        const cameraModal = document.getElementById('cameraModal');
        const cameraVideo = document.getElementById('cameraVideo');
        const cameraCanvas = document.getElementById('cameraCanvas');
        const captureBtn = document.getElementById('captureBtn');
        const closeCameraBtn = document.getElementById('closeCameraBtn');

        // Touch-friendly upload area
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // Enhanced drag and drop with proper file handling
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                selectedFile = files[0];
                handleFile(files[0]);
            }
        });

        // File input change handler
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                handleFile(e.target.files[0]);
            }
        });

        // Handle file selection
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('Please select a valid image file (JPG, PNG, JPEG)');
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                imagePreview.style.display = 'block';
                emptyState.style.display = 'none';
                analyzeBtn.disabled = false;
                hideError();
                hideResult();
            };
            reader.readAsDataURL(file);
        }

        // Camera functionality
        cameraBtn.addEventListener('click', async () => {
            try {
                let constraints = { video: true };
                
                try {
                    stream = await navigator.mediaDevices.getUserMedia(constraints);
                } catch (basicError) {
                    constraints = { 
                        video: { 
                            facingMode: { ideal: 'environment' },
                            width: { min: 640, ideal: 1280 },
                            height: { min: 480, ideal: 720 }
                        } 
                    };
                    stream = await navigator.mediaDevices.getUserMedia(constraints);
                }
                
                cameraVideo.srcObject = stream;
                cameraModal.style.display = 'block';
            } catch (err) {
                console.error('Camera error:', err);
                showError('Camera access denied. Please enable camera permissions in browser settings and refresh.');
            }
        });

        captureBtn.addEventListener('click', () => {
            const canvas = cameraCanvas;
            const context = canvas.getContext('2d');
            canvas.width = cameraVideo.videoWidth;
            canvas.height = cameraVideo.videoHeight;
            context.drawImage(cameraVideo, 0, 0);
            
            canvas.toBlob((blob) => {
                selectedFile = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                previewImg.src = url;
                imagePreview.style.display = 'block';
                emptyState.style.display = 'none';
                analyzeBtn.disabled = false;
                closeCamera();
                hideError();
                hideResult();
            }, 'image/jpeg', 0.9);
        });

        closeCameraBtn.addEventListener('click', closeCamera);

        function closeCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            cameraModal.style.display = 'none';
        }

        // Close camera modal when clicking outside
        cameraModal.addEventListener('click', (e) => {
            if (e.target === cameraModal) {
                closeCamera();
            }
        });

        // Analyze button click handler
        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                showError('Please select an image first');
                return;
            }

            showLoading();
            hideError();
            hideResult();

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    showResult(data);
                } else {
                    showError(data.error || 'Neural analysis failed');
                }
            } catch (err) {
                showError('Failed to connect to NeuroAI service. Ensure the system is running.');
            } finally {
                hideLoading();
            }
        });

        // Show loading state
        function showLoading() {
            loading.style.display = 'block';
            analyzeBtn.disabled = true;
        }

        // Hide loading state
        function hideLoading() {
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
        }

        // Show prediction results with enhanced layout
        function showResult(data) {
            const prediction = data.predicted_class.replace(/_/g, ' ').toUpperCase();
            const confidence = (data.confidence * 100).toFixed(1) + '%';
            const confidenceLevel = data.confidence_level || 'Unknown';
            
            // Update main prediction display
            document.getElementById('predictionValue').textContent = prediction;
            document.getElementById('confidenceValue').textContent = `Confidence: ${confidence}`;
            document.getElementById('confidenceLevel').textContent = `Level: ${confidenceLevel}`;
            
            // Show detailed probabilities in grid layout
            const probabilityGrid = document.getElementById('probabilityGrid');
            probabilityGrid.innerHTML = '';
            
            // Sort probabilities by value for better visualization
            const sortedProbs = Object.entries(data.all_probabilities)
                .sort(([,a], [,b]) => b - a);
            
            sortedProbs.forEach(([className, probability], index) => {
                const cleanName = className.replace(/_/g, ' ').toUpperCase();
                const percentValue = (probability * 100).toFixed(1);
                
                const card = document.createElement('div');
                card.className = 'probability-card';
                card.style.animationDelay = `${index * 0.1}s`;
                card.innerHTML = `
                    <div class="probability-name">${cleanName}</div>
                    <div class="circular-progress">
                        <svg viewBox="0 0 100 100">
                            <circle class="track" cx="50" cy="50" r="40"></circle>
                            <circle class="progress" cx="50" cy="50" r="40" data-percent="${percentValue}"></circle>
                        </svg>
                        <div class="percentage">${Math.round(percentValue)}%</div>
                    </div>
                `;
                probabilityGrid.appendChild(card);
                
                // Animate the circular progress
                setTimeout(() => {
                    const progressCircle = card.querySelector('.progress');
                    const circumference = 2 * Math.PI * 40;
                    const offset = circumference - (percentValue / 100) * circumference;
                    progressCircle.style.strokeDashoffset = offset;
                }, 500 + (index * 150));
            });
            
            // Show results container with smooth scroll
            resultsContainer.style.display = 'block';
            setTimeout(() => {
                resultsContainer.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                });
            }, 300);
        }

        // Show error message
        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
        }

        // Hide error message
        function hideError() {
            error.style.display = 'none';
        }

        // Hide result
        function hideResult() {
            resultsContainer.style.display = 'none';
        }

        // Reset application
        function resetApp() {
            selectedFile = null;
            fileInput.value = '';
            imagePreview.style.display = 'none';
            emptyState.style.display = 'block';
            analyzeBtn.disabled = true;
            hideResult();
            hideError();
            hideLoading();
            
            // Smooth scroll back to top
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        // Add CSS animation for probability cards
        const style = document.createElement('style');
        style.textContent = `            
            .probability-card {
                animation: slideInUp 0.6s ease-out forwards;
                opacity: 0;
                transform: translateY(30px);
            }
            
            @keyframes slideInUp {
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .results-container {
                animation: slideInUp 0.8s ease-out;
            }
        `;
        document.head.appendChild(style);

        // Test API connection on page load
        window.addEventListener('load', async () => {
            try {
                const response = await fetch(window.location.origin + '/health');
                if (response.ok) {
                    console.log('✅ NeuroAI System Online');
                }
            } catch (err) {
                console.log('⚠️ NeuroAI System Offline');
            }
        });

        // Prevent zoom on double tap for iOS
        let lastTouchEnd = 0;
        document.addEventListener('touchend', (event) => {
            const now = (new Date()).getTime();
            if (now - lastTouchEnd <= 300) {
                event.preventDefault();
            }
            lastTouchEnd = now;
        }, false);

        // Add haptic feedback for supported devices
        function hapticFeedback() {
            if ('vibrate' in navigator) {
                navigator.vibrate(50);
            }
        }

        // Add haptic feedback to buttons
        analyzeBtn.addEventListener('click', hapticFeedback);
        cameraBtn.addEventListener('click', hapticFeedback);
        captureBtn.addEventListener('click', hapticFeedback);

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'u':
                        e.preventDefault();
                        fileInput.click();
                        break;
                    case 'Enter':
                        e.preventDefault();
                        if (!analyzeBtn.disabled) {
                            analyzeBtn.click();
                        }
                        break;
                    case 'r':
                        e.preventDefault();
                        resetApp();
                        break;
                }
            }
        });

        // Add visual feedback for touch interactions
        document.addEventListener('touchstart', (e) => {
            if (e.target.matches('button, .upload-area')) {
                e.target.style.transform = 'scale(0.95)';
            }
        });

        document.addEventListener('touchend', (e) => {
            if (e.target.matches('button, .upload-area')) {
                setTimeout(() => {
                    e.target.style.transform = '';
                }, 150);
            }
        });
    </script>
</body>
</html>'''

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'predictor_loaded': predictor is not None,
        'message': 'NeuroAI System Online'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with robust error handling"""
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'AI model not loaded. Please restart the system.'
            }), 500

        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload an image file.'
            }), 400

        # Process the prediction
        try:
            # Use the robust predictor with the uploaded file
            result = predictor.predict(file)
            
            # Check if prediction was successful
            if 'error' in result:
                return jsonify({
                    'success': False,
                    'error': f'Prediction failed: {result["error"]}'
                }), 500

            # Format the response to match the frontend expectations
            response_data = {
                'success': True,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'confidence_level': result['confidence_level'],
                'is_reliable': result['is_reliable'],
                'all_probabilities': result['all_probabilities'],
                'processing_time': time.time() - time.time(),  # You can add actual timing if needed
                'model_info': {
                    'version': 'Advanced v2.0',
                    'accuracy': '95%+',
                    'classes': predictor.class_names
                }
            }
            
            return jsonify(response_data)

        except Exception as pred_error:
            print(f"Prediction error: {pred_error}")
            return jsonify({
                'success': False,
                'error': f'Prediction processing failed: {str(pred_error)}'
            }), 500

    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error occurred'
        }), 500

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    if predictor is None:
        return jsonify({
            'loaded': False,
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'loaded': True,
        'classes': predictor.class_names,
        'confidence_threshold': predictor.confidence_threshold,
        'model_type': 'Robust Brain Tumor Classifier',
        'version': '2.0',
        'preprocessing': 'Advanced with CLAHE, noise reduction, and smart cropping'
    })

if __name__ == '__main__':
    print("🚀 Starting NeuroAI Advanced Brain Tumor Detection System...")
    print("🔗 Access the application at: http://localhost:5005")
    print("📊 Model loaded:", "✅ Yes" if predictor else "❌ No")
    print("🧠 Advanced preprocessing enabled")
    print("=" * 60)
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5005,
        threaded=True
    )