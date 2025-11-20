from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from flask_mail import Mail, Message
from .models import db, User
from werkzeug.security import generate_password_hash, check_password_hash
import random
import string
import os

bp = Blueprint('app2', __name__, template_folder='templates')

# Initialize Flask-Mail
mail = Mail()

@bp.record
def record_params(setup_state):
    app = setup_state.app
    mail.init_app(app)
    with app.app_context():
        db.create_all()  # Create tables for new database

def generate_verification_code(length=6):
    """Generate a random verification code"""
    return ''.join(random.choices(string.digits, k=length))

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            if not user.is_verified:
                flash('Please verify your email first')
                return redirect(url_for('app2.verify_email'))
            session['user_id'] = user.id
            return redirect(url_for('app1.home'))
        flash('Invalid credentials')
    
    return render_template('login.html')

@bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return render_template('signup.html')
        
        verification_code = generate_verification_code()
        user = User(
            name=name,
            email=email,
            password=generate_password_hash(password),
            verification_code=verification_code,
            is_verified=False
        )
        db.session.add(user)
        db.session.commit()
        
        # Send verification email
        msg = Message('Verify Your Email',
                     sender=os.getenv('MAIL_DEFAULT_SENDER'),
                     recipients=[email])
        msg.body = f'Your verification code is: {verification_code}'
        try:
            mail.send(msg)
            session['user_id'] = user.id
            return redirect(url_for('app2.verify_email'))
        except Exception as e:
            flash('Error sending verification email')
            db.session.delete(user)
            db.session.commit()
            return render_template('signup.html')
    
    return render_template('signup.html')

@bp.route('/verify_email', methods=['GET', 'POST'])
def verify_email():
    if 'user_id' not in session:
        return redirect(url_for('app2.login'))
        
    user = User.query.get(session['user_id'])
    if not user:
        session.pop('user_id', None)
        return redirect(url_for('app2.login'))
        
    if request.method == 'POST':
        code = request.form.get('code')
        if code == user.verification_code:
            user.is_verified = True
            user.verification_code = None
            db.session.commit()
            return redirect(url_for('app1.home'))
        flash('Invalid verification code')
    
    return render_template('verify.html')

@bp.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('app2.login'))