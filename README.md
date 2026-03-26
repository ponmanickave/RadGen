# 🩺 RadGen: Multimodal Neural Vision Engine

RadGen is an advanced AI-powered medical diagnostic platform designed to analyze chest X-ray images and generate automated, structured medical reports. Using deep learning (ResNet50 + Transformers), it assists radiologists by providing rapid initial findings, improving diagnostic efficiency.

---

## 🚀 Key Features

*   **AI-Powered Analysis**: Uses a ResNet50-based CNN to extract features from X-ray images and a multimodal transformer model to generate findings.
*   **Structured Medical Reports**: Automatically generates detailed findings for lungs, heart, pleura, and osseous structures.
*   **User Management**: Secure login and registration for doctors and medical staff (Roles: Doctor, Admin).
*   **Dashboard & Archive**: Track patient reports, view history, and manage medical records in a clean, JSON-based database.
*   **PDF Generation & Sharing**: Instantly generate high-quality PDF reports and share them directly via email.
*   **Automated Reminders**: Built-in scheduler to send follow-up reminders for patients with "Malignant" findings.

---

## 🛠️ Technology Stack

*   **Backend**: Python, Flask, Flask-Login, Flask-Mail, APScheduler
*   **AI/ML**: TensorFlow, NumPy, ResNet50
*   **Reporting**: ReportLab (PDF Generation)
*   **Frontend**: HTML5, Vanilla CSS3 (Modern, Responsive Dashboard)
*   **Database**: JSON (Lightweight local storage)

---

## 📂 Project Structure

```text
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── tokenizer.json        # Pre-trained text tokenizer
├── max_sequence_length.txt # Model configuration
├── utils/                # Helper modules
│   ├── model_handler.py  # AI Model loading & inference
│   └── pdf_generator.py  # PDF report styling & generation
├── templates/            # HTML layouts
├── static/               # CSS, JS, and UI Assets
├── codes/                # Machine Learning training scripts
└── database.json         # Local user and report database
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/RadGen1.git
cd RadGen1
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt
```

### 3. Model Files (IMPORTANT)
The trained AI model (`multimodal_report_generator.h5`) and embedding files (`.npy`) are too large for GitHub (over 100MB). 
*   **To run the AI**: You must place `multimodal_report_generator.h5` in the project root.
*   *Note:* If you are hosting on Render or Netlify without the model, the app will run in "Demo Mode" only.

### 4. Run the Application
```bash
python app.py
```
Visit `http://127.0.0.1:5001` in your browser.

---

## 📧 Email Configuration
To enable the "Share Report via Email" feature, go to `app.py` and update the following configuration with your SMTP credentials:
```python
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your-app-password'
```

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

---

## 👥 Contributors
Developed by **K.M.Dhoni** as a Final Year Project on Neural Vision Systems.
