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


def _send_welcome_email(user):
    msg = Message(
        'Welcome to AURA AI',
        sender=os.getenv('MAIL_DEFAULT_SENDER'),
        recipients=[user.email]
    )
    msg.body = (
        f"Hi {user.name},\n\n"
        "Welcome to AURA AI. Your account is verified and you can now access your "
        "AI-powered radiology workspace. Explore your dashboard, review past "
        "studies, and upload new scans with a single click.\n\n"
        "Next steps:\n"
        "1. Log in to your workspace to personalize notification settings.\n"
        "2. Review the quick-start guide in the Help panel to learn how to run "
        "inference, compare versions, and export reports.\n"
        "3. Reach out to our clinical success team if you would like a curated "
        "walkthrough for your practice.\n\n"
        "If you did not create this account, please contact support immediately so "
        "we can secure your information."
    )
    msg.html = (
        "<div style=\"font-family:Arial,sans-serif; background:#070605; color:#f4f2ef; "
        "padding:24px; border-radius:12px;\">"
        f"<h2 style=\"margin:0 0 12px; font-weight:600;\">Welcome, {user.name}</h2>"
        "<p style=\"margin:0 0 16px; color:#d7d2cc;\">"
        "Your account is verified and you can now access your AURA AI workspace. "
        "Review historical studies, upload new images, and generate explainable "
        "diagnostic overlays in seconds."
        "</p>"
        "<ul style=\"margin:0 0 16px 18px; padding:0; color:#bfb8ae; font-size:14px;\">"
        "<li>Personalize alerts and notifications</li>"
        "<li>Follow the quick-start checklist in the Help panel</li>"
        "<li>Invite teammates for collaborative case review</li>"
        "</ul>"
        "<p style=\"margin:0; color:#a69f97; font-size:13px;\">"
        "If you did not create this account, please contact support immediately so we can secure your workspace."
        "</p>"
        "</div>"
    )
    mail.send(msg)

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
        msg.body = (
            "Welcome to AURA AI.\n\n"
            "Use the verification code below to complete your sign-up. "
            "This code expires soon, so please enter it promptly. Once you are "
            "verified, you can explore guided workflows, upload clinical images, "
            "and generate transparent explainability maps.\n\n"
            f"Verification code: {verification_code}\n\n"
            "Need help? Open the Help panel after logging in or reply to this "
            "message to reach our clinical success team.\n\n"
            "If you did not create this account, you can ignore this email and "
            "the request will expire automatically."
        )
        msg.html = (
            "<div style=\"font-family:Arial,sans-serif; background:#070605; color:#f4f2ef; "
            "padding:24px; border-radius:12px;\">"
            "<h2 style=\"margin:0 0 12px; font-weight:600;\">Welcome to AURA AI</h2>"
            "<p style=\"margin:0 0 16px; color:#d7d2cc;\">"
            "Use the verification code below to complete your sign-up. This code expires soon. "
            "After verification you will gain access to guided workflows, rapid uploads, and Grad-CAM explainability views."
            "</p>"
            "<div style=\"font-size:32px; font-weight:700; letter-spacing:6px; color:#b28a5b; "
            "margin:12px 0 20px;\">"
            f"{verification_code}"
            "</div>"
            "<p style=\"margin:0 0 12px; color:#bfb8ae; font-size:14px;\">"
            "Need help verifying? Reply to this email or open the Help panel in the app for a guided walkthrough."
            "</p>"
            "<p style=\"margin:0; color:#a69f97; font-size:13px;\">"
            "If you did not create this account, you can safely ignore this email and the request will expire."
            "</p>"
            "</div>"
        )
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
        code_parts = request.form.getlist('code[]')
        if code_parts:
            code = ''.join(code_parts).strip()
        else:
            code = (request.form.get('code') or '').strip()

        if code and code == user.verification_code:
            user.is_verified = True
            user.verification_code = None
            db.session.commit()
            try:
                _send_welcome_email(user)
            except Exception:
                flash('Your account was verified, but we could not send the welcome email.')
            return redirect(url_for('app1.home'))
        flash('Invalid verification code')
    
    return render_template('verify.html')

@bp.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('app2.login'))