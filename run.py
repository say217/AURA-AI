from app1 import create_app as create_app1
from app2 import create_app as create_app2
from app3 import create_app as create_app3
from app4 import create_app as create_app4
from app5 import create_app as create_app5
from flask import Flask, redirect, url_for, session
from flask_mail import Mail
from flask_migrate import Migrate
from app2.models import db  
import os
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)


# Configuration settingsjupyter notebook
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aura.db'  # Uses aura.db
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')  # Fallback if not set
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True

app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

# Initialize Flask-Mail and Flask-Migrate
Mail(app)
db.init_app(app)  # Initialize db with the app
migrate = Migrate(app, db)  # Initialize Flask-Migrate

# Register blueprints
app.register_blueprint(create_app1(), url_prefix='/app1')
app.register_blueprint(create_app2(), url_prefix='/app2')
app.register_blueprint(create_app3(), url_prefix='/app3')
app.register_blueprint(create_app4(), url_prefix='/app4')
app.register_blueprint(create_app5(), url_prefix='/app5')

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('app1.home'))
    return redirect(url_for('app2.login'))

if __name__ == '__main__':
    app.run(debug=True)