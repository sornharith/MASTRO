"""
Setup script for the MASTRO (Multi-Agent System for Time-series Dropout Prediction that integrates specialized learning agents and reasoning mechanisms within a unified coordination framework)
"""
from setuptools import setup, find_packages

setup(
    name="MASTRO",
    version="1.0.0",
    description="Multi-Agent System for Time-series Dropout Prediction that integrates specialized learning agents and reasoning mechanisms within a unified coordination framework",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "scikit-learn",
        "catboost",
        "optuna",
        "transformers",
        "peft",
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "tqdm",
        "shap",
        "joblib",
        "accelerate",
        "bitsandbytes"
    ],
    python_requires=">=3.10",
)