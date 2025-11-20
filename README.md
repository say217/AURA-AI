# Aura AI – AI Unified Radiology Assistant
A Modular Flask-Based Web Application for AI-Powered Cancer Detection in Histopathology Images

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/7110d791-aea4-4cea-af22-4f016bd2ea7f" alt="Image 1" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/463d88bb-4f0d-4788-b2b1-5ad68258ab79" alt="Image 2" width="400"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/11b4fef7-b07f-4301-81da-01d748b877b5" alt="Image 3" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/9ac215a9-4101-4956-a68b-70d9f9336dc3" alt="Image 4" width="400"/></td>
  </tr>
</table>

# Introduction

**Aura AI – Unified Radiology Assistant** is a next-generation, AI-powered diagnostic support system designed to assist pathologists, oncology researchers, and medical students in the rapid preliminary assessment of histopathology images. Leveraging deep learning (**PyTorch + ResNet18**) and explainable AI (**Grad-CAM**), Aura AI provides both accurate predictions and transparent visual reasoning—offering insights into why the model makes a particular decision.

Developed as a modular **Flask** application using the *Application Factory + Blueprint* pattern, the platform is scalable, maintainable, and easily extendable with new cancer models or diagnostic modules. Aura AI currently supports:

- **Breast Cancer – IDC Detection (2-class)**
- **Lung & Colon Cancer – 5-class tissue classification**

This system is intended for **research and educational use**, providing an intuitive interface, secure authentication, and clinically informed visual explanations.

# Project Overview & Purpose

**Aura AI** is an advanced, user-centric histopathology analysis assistant built to support fast, reliable, and explainable assessment of cancerous tissues. The system streamlines preliminary diagnostic screening for two major cancer families—**Breast Cancer** and **Lung & Colon Cancer**—while combining deep learning, pathology-aligned workflows, and explainable AI to create a powerful tool for researchers, clinicians, and medical students.

Aura AI focuses on delivering actionable insights with clarity, using well-structured predictions, heatmaps, probability visualizations, and clinical text interpretations. Its modular design ensures long-term scalability, allowing new cancer types or entire diagnostic pipelines to be integrated with minimal friction.

---

## 🟣 Breast Cancer (App4 — Primary Module)

The breast cancer module serves as one of the platform’s core diagnostic components. It performs **binary classification** to differentiate between:

- **IDC+ (Invasive Ductal Carcinoma Positive)**
- **IDC− (Invasive Ductal Carcinoma Negative)**

To enhance user understanding and diagnostic confidence, each prediction is paired with:

- **Grad-CAM heatmaps** that highlight regions most influential to the model’s decision  
- **Clinical interpretation text** that contextualizes the prediction  
- **Multi-image / batch processing**, enabling users to screen multiple slides efficiently  

This module is primarily designed for rapid IDC screening and educational visualization.

---

## Lung & Colon Cancer (App3)

The Lung & Colon module provides a more diverse classification system, offering **5-class tissue categorization** to help distinguish between malignant and benign cases across two organ systems:

- Lung Adenocarcinoma  
- Lung Squamous Cell Carcinoma  
- Lung Benign  
- Colon Adenocarcinoma  
- Colon Benign  

Each prediction is supported by visual analytics, including:

- **Probability bar charts** for intuitive comparison of model confidence across all classes  
- **Grad-CAM heatmaps** that reveal high-impact tissue regions  

This broader classification supports research use cases where tissue differentiation plays a critical role.

---
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c9e1eb67-8b00-4a10-85ad-0fe44090416e" alt="Image 1" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/53df5dc6-0f8e-4045-9753-077bfa748b2f" alt="Image 2" width="400"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/1876c1f8-6879-4192-be89-e76800ab99d0" alt="Image 3" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/4e111951-25ca-49f2-ac89-27e3f8592cb1" alt="Image 4" width="400"/></td>
  </tr>
</table>

---

## Explainable AI (XAI) Integration
Custom-built Grad-CAM implementation from scratch (no black-box models). Generates high-resolution, color-coded heatmaps superimposed on original microscopy images, enabling medical professionals to understand and trust every prediction.


## Multiple & Batch Image Upload
Seamless drag-and-drop or select multiple histopathological images at once (supported formats: .png, .jpg, .jpeg, .tif). Real-time batch inference with individual results and visualizations returned instantly.

---

## Clean, Modern & Responsive UI
A polished, medical-grade interface built with **Bootstrap 5** featuring:

- Mobile-first responsive layout  
- Intuitive navigation  
- Dark/Light mode support  
- Professional styling suitable for clinical and academic environments  

---

##  Scalable & Production-Ready Architecture
Engineered using a **modular Flask Blueprint architecture** ensuring:

- Clean separation of application modules  
- Environment-variable configuration (`.env`)  
- Gunicorn + Nginx compatibility  
- Integrated logging and error handling  
- Docker-ready deployment structure  
Modular Flask application using Blueprints for clean separation of concerns, configuration via environment variables (.env), Gunicorn + Nginx ready, logging, error handling, and structured for easy deployment (Docker support included).


## Tech Stack
| Layer                 | Technology                                                                        |
| --------------------- | --------------------------------------------------------------------------------- |
| **Backend Framework** | Flask 3.x, Flask-SQLAlchemy, Flask-Migrate, Flask-Mail                            |
| **Frontend**          | HTML5, CSS3, Jinja2 templating, Vanilla JS
| **Deep Learning**     | PyTorch 2.x, torchvision, OpenCV, NumPy, Matplotlib/Seaborn                       |
| **Database**          | SQLite (dev) → easy migration to PostgreSQL/MySQL                                 |
| **Authentication**    | Flask sessions + bcrypt hashing                                                   |
| **Explainability**    | Custom Grad-CAM implementation                                                    |
| **Deployment Ready**  | `.env` support, Gunicorn-compatible, requirements.txt                             |


```
aura-ai/
├── app1/                  # Home/Dashboard (after login)
│   ├── templates/home.html
│   ├── routes.py
│   └── __init__.py
│
├── app2/                  # Authentication Blueprint
│   ├── templates/
│   │   ├── login.html
│   │   ├── signup.html
│   │   └── verify.html
│   ├── routes.py
│   ├── models.py          # User model + DB
│   └── __init__.py
│
├── app3/                  # Lung & Colon 5-class Detection
│   ├── templates/home2.html
│   ├── routes.py
│   ├── resnet18_model_001.pth
│   └── __init__.py
│
├── app4/                  # Breast Cancer IDC Detection
│   ├── templates/home3.html
│   ├── routes.py
│   ├── static/uploads/    # Generated CAM heatmaps
│   ├── breast_cancer_cnn_model_updated.pth
│   └── __init__.py
│
├── instance/              # SQLite DB (gitignored)
├── images/                # Screenshots
├── static/                # Global CSS/JS
├── templates/             # Global base.html
├── .env                   # Environment variables
├── .gitignore
├── requirements.txt
├── run.py                 # Application entry point
└── README.md


```
Clone the repo
```
git clone https://github.com/yourusername/aura-ai-radiology-assistant.git
cd aura-ai-radiology-assistant
```
Create Virtual Environment
```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```
Install dependencies
```
pip install -r requirements.txt
```
Create .env file
```txt
SECRET_KEY=your_random_secret_key
MAIL_USERNAME=yourgmail@gmail.com
MAIL_PASSWORD=your_app_password
MAIL_DEFAULT_SENDER=yourgmail@gmail.com
```
Initialize database
```
flask db init
flask db migrate
flask db upgrade
```
Run the app
```
python run.py
# or
flask run
```

### Future Enhancements (Roadmap)

Add additional cancer types (Prostate, Kidney, Brain)
Dockerization (Dockerfile + Compose)
Switch to PostgreSQL + Redis for production
Automated PDF report generation
Role-based access (Doctor / Patient)
Admin panel + model versioning
Cloud deployment (Render, Railway, AWS)

## Important Disclaimer

Aura AI is intended for research and educational use only.
This system must not be used for clinical diagnosis.
All predictions must be validated by a certified pathologist or oncologist.

## Contribution Guidelines

- Fork the repository
- Create a feature branch
```
git checkout -b feature/my-feature
```
- Commit changes
- Push to your branch
- Open a Pull Request

## Acknowledgements

Aura AI is built upon the contributions of the open-source community. Special thanks to:
The creators and maintainers of PyTorch, Flask, Bootstrap, and associated libraries
Researchers and dataset providers behind LC25000, NCT-CRC-HE, and the IDC Breast Cancer dataset
Medical professionals whose insights helped shape clinical interpretation features. Aura AI represents an important step toward making AI tools more transparent, accessible, and clinically meaningful. While the system is not intended for real-world medical diagnosis, it demonstrates how modern deep learning and explainable AI methods can support and augment decision-making in pathology.

