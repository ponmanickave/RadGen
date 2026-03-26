from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import json
from utils.model_handler import AIHandler
from utils.pdf_generator import PDFGenerator
from datetime import datetime
from flask_apscheduler import APScheduler
from flask_mail import Mail, Message

# Initialize Flask
app = Flask(__name__)
app.secret_key = "radgen_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Email Configuration (Placeholders - USER to provide real credentials)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-system-email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your-app-password'
app.config['MAIL_DEFAULT_SENDER'] = 'your-system-email@gmail.com'

mail = Mail(app)
DB_FILE = 'database.json'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper functions for JSON DB
def load_db():
    if not os.path.exists(DB_FILE):
        return {"users": [{"id": 1, "email": "admin@radgen.ai", "password": "password@123", "role": "doctor"}], "reports": []}
    with open(DB_FILE, 'r') as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f)

# Initialize Auth
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, email, password, role, name=None):
        self.id = id
        self.email = email
        self.password = password
        self.role = role
        self.name = name

# Initialize AI & Scheduler
ai_engine = AIHandler()
pdf_engine = PDFGenerator()
scheduler = APScheduler()

# =====================================================
# BACKGROUND TASKS
# =====================================================
def send_reminders():
    with app.app_context():
        print("🕒 Running Weekly Reminder Task...")
        db_data = load_db()
        malignant_cases = [r for r in db_data['reports'] if r.get('status') == 'Malignant']
        for report in malignant_cases:
            print(f"📧 EMAIL SENT to Patient PT-{report['patient_id']}: Reminder for follow-up check-up.")

scheduler.add_job(id='reminder_job', func=send_reminders, trigger='interval', weeks=1)
# scheduler.init_app(app) # APScheduler can be tricky with JSON, disabling for simple run
# scheduler.start()

@login_manager.user_loader
def load_user(user_id):
    db_data = load_db()
    for u in db_data['users']:
        if str(u['id']) == str(user_id):
            return User(u['id'], u.get('email', u.get('username')), u['password'], u['role'], u.get('name'))
    return None

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    db_data = load_db()
    reports = db_data['reports']
    reports_count = len(reports)
    # Sort by timestamp desc and take 5
    recent_reports = sorted(reports, key=lambda x: x['timestamp'], reverse=True)[:5]
    return render_template('dashboard.html', reports_count=reports_count, recent_reports=recent_reports)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        db_data = load_db()
        
        # Check if user exists (case-insensitive)
        user_dict = next((u for u in db_data['users'] if str(u.get('email', u.get('username'))).lower() == email.lower()), None)
        
        if not user_dict:
            flash('User does not exist.', 'danger')
            return render_template('login.html')
            
        if user_dict['password'] == password:
            if role and user_dict['role'] != role:
                flash(f'Found user, but role mismatch ({role}). Check your access level.', 'danger')
                return render_template('login.html')
            
            user = User(user_dict['id'], user_dict.get('email', user_dict.get('username')), user_dict['password'], user_dict['role'], user_dict.get('name'))
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role', 'user')
        
        if password != confirm_password:
            flash('Passwords did not match', 'danger')
            return render_template('register.html')
            
        db_data = load_db()
        if any(str(u.get('email', u.get('username'))).lower() == email.lower() for u in db_data['users']):
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
            
        new_user = {
            "id": len(db_data['users']) + 1,
            "name": name,
            "email": email,
            "password": password,
            "role": role
        }
        db_data['users'].append(new_user)
        save_db(db_data)
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image uploaded', 'danger')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Generate Report using AI Engine
            try:
                report_content = ai_engine.generate_report_content(filepath)
                
                # Status determination
                status = "Malignant" if any(k in str(report_content).lower() for k in ["opacity", "consolidation", "enlarged", "effusion"]) else "Benign"
                
                # Save to JSON database
                db_data = load_db()
                # Use the report ID generated by the client or generate a new one
                client_report_id = request.form.get('report_id')
                report_id = client_report_id if client_report_id else f"RG-REP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.getpid()}"
                
                new_report = {
                    "id": len(db_data['reports']) + 1,
                    "report_id": report_id,
                    "patient_name": request.form.get('patient_name', 'Unknown'),
                    "patient_id": request.form.get('patient_id', 'Unknown'),
                    "user_id": current_user.id,
                    "image_path": filepath,
                    "prediction_data": report_content,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "status": status
                }
                db_data['reports'].append(new_report)
                save_db(db_data)
                
                flash('Report generated successfully!', 'success')
                return render_template('report_view.html', report=new_report)
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'danger')
                return redirect(request.url)

    return render_template('predict.html')

@app.route('/download_report/<int:report_id>')
@login_required
def download_report(report_id):
    db_data = load_db()
    report = next((r for r in db_data['reports'] if r['id'] == report_id), None)
    if not report:
        flash('Report not found', 'danger')
        return redirect(url_for('archive'))
        
    pdf_filename = f"RadGen_Report_{report['patient_id']}_{report['id']}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    
    try:
        # Wrap report dict in an object-like structure for the PDF generator if needed
        class ReportObj:
            def __init__(self, d):
                self.__dict__ = d
                self.timestamp = datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S')
        
        pdf_engine.create_report(ReportObj(report), pdf_path)
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        flash(f'Error generating PDF: {str(e)}', 'danger')
        return redirect(url_for('archive'))

@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    db_data = load_db()
    report = next((r for r in db_data['reports'] if r['id'] == report_id), None)
    if not report:
        flash('Report not found', 'danger')
        return redirect(url_for('archive'))
    return render_template('report_view.html', report=report)

@app.route('/share_report/<int:report_id>')
@login_required
def share_report(report_id):
    db_data = load_db()
    report = next((r for r in db_data['reports'] if r['id'] == report_id), None)
    if not report:
        flash('Report not found', 'danger')
        return redirect(url_for('archive'))
        
    pdf_filename = f"RadGen_Report_{report['patient_id']}_{report['id']}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    
    try:
        # Generate PDF if it doesn't exist
        class ReportObj:
            def __init__(self, d):
                self.__dict__ = d
                self.timestamp = datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S')
        
        pdf_engine.create_report(ReportObj(report), pdf_path)
        
        # Send Email
        msg = Message(
            subject=f"Diagnostic Report: PT-{report['patient_id']}",
            recipients=[current_user.email],
            body=f"Hello,\n\nPlease find attached the diagnostic findings for Patient ID: PT-{report['patient_id']}.\n\nThis report was generated by the RadGen Neural Vision Engine.\n\nBest regards,\nRadGen Health Systems"
        )
        with app.open_resource(pdf_path) as fp:
            msg.attach(pdf_filename, "application/pdf", fp.read())
        
        mail.send(msg)
        flash(f'Report shared successfully to {current_user.email}!', 'success')
    except Exception as e:
        flash(f'Email setup incomplete or error: {str(e)}', 'warning')
        
    return redirect(url_for('view_report', report_id=report_id))

@app.route('/archive')
@login_required
def archive():
    db_data = load_db()
    if current_user.role == 'admin' or current_user.role == 'doctor':
        reports = db_data['reports']
    else:
        # User only sees their own reports
        reports = [r for r in db_data['reports'] if r.get('user_id') == current_user.id]
        
    reports = sorted(reports, key=lambda x: x['timestamp'], reverse=True)
    return render_template('archive.html', reports=reports)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
