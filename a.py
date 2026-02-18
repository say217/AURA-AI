import pkg_resources

modules = [
    "flask",
    "flask-mail",
    "flask-migrate",
    "flask_sqlalchemy",
    "python-dotenv",
    "torch",
    "torchvision",
    "Pillow",
    "numpy",
    "opencv-python",
    "matplotlib",
    "seaborn",
    "werkzeug",
    "itsdangerous",
    "passlib"
]

print("\n=== INSTALLED MODULE VERSIONS ===\n")
for m in modules:
    try:
        version = pkg_resources.get_distribution(m).version
        print(f"{m:20}  ->  {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{m:20}  ->  NOT INSTALLED")

print("\nDone.\n")
