from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from .database import SessionLocal, engine, create_tables
from .models import Base, LoanApplication, User, ModelMetrics
from .schemas import ApplicationCreate, ApplicationUpdate, UserCreate, UserLogin, ModelMetricsCreate
from .auth import get_password_hash, authenticate_user
from .config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, ADMIN_USERNAME, ADMIN_PASSWORD
from datetime import timedelta, datetime
from jose import jwt, JWTError
from fastapi.responses import FileResponse  
from .database import create_tables


# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI
app = FastAPI()

# # Mount static files
# app.mount("/static", StaticFiles(directory="static"), name="static")
# # In app/main.py, replace this line:
# app.mount("/static", StaticFiles(directory="static"), name="static")

# With:
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "..", "static")), name="static")
# Initialize templates
templates = Jinja2Templates(directory="templates")
templates.env.globals['current_year'] = datetime.utcnow().year

# Load ML model
try:
    model = joblib.load("app/ml/model.pkl")
    scaler = joblib.load("app/ml/scaler.pkl")
    feature_names = joblib.load("app/ml/feature_names.pkl")
    metrics = joblib.load("app/ml/metrics.pkl")
    rf_metrics = joblib.load("app/ml/random_forest_metrics.pkl")
except Exception as e:
    print(f"Error loading ML artifacts: {e}")
    raise

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize admin user and metrics
def init_admin_user_and_metrics():
    db = SessionLocal()
    try:
        # Create admin user
        admin = db.query(User).filter(User.username == ADMIN_USERNAME).first()
        if not admin:
            hashed_password = get_password_hash(ADMIN_PASSWORD)
            admin_user = User(
                username=ADMIN_USERNAME,
                password=hashed_password,
                is_admin=True
            )
            db.add(admin_user)
        
        # Store metrics in database
        if not db.query(ModelMetrics).first():
            best_metrics = ModelMetrics(
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1'],
                rf_accuracy=rf_metrics['accuracy'],
                rf_precision=rf_metrics['precision'],
                rf_recall=rf_metrics['recall'],
                rf_f1=rf_metrics['f1'],
                last_updated=datetime.utcnow()
            )
            db.add(best_metrics)
            db.commit()
    except Exception as e:
        print(f"Error initializing admin user and metrics: {e}")
        db.rollback()
    finally:
        db.close()

# Preprocess application data for prediction
def preprocess_application_data(data: dict):
    # Create DataFrame from input data
    df = pd.DataFrame([data])
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Ensure all expected columns are present
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns
    df_encoded = df_encoded[feature_names]
    
    # Scale the data
    scaled_data = scaler.transform(df_encoded)
    return scaled_data

# Create access token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Get current user from token
def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    
    user = db.query(User).filter(User.username == username).first()
    return user

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Apply for loan form
@app.get("/apply", response_class=HTMLResponse)
async def apply_form(request: Request):
    return templates.TemplateResponse("apply.html", {"request": request})

# Process loan application
@app.post("/apply", response_class=HTMLResponse)
async def apply_loan(
    request: Request,
    db: Session = Depends(get_db),
    applicant_name: str = Form(...),
    gender: str = Form(...),
    married: str = Form(...),
    dependents: str = Form(...),
    education: str = Form(...),
    self_employed: str = Form(...),
    applicant_income: float = Form(...),
    coapplicant_income: float = Form(...),
    loan_amount: float = Form(...),
    loan_amount_term: float = Form(...),
    credit_history: float = Form(...),
    property_area: str = Form(...),
):
    # Prepare data for prediction
    form_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }
    
    # Preprocess and predict
    processed_data = preprocess_application_data(form_data)
    prediction = model.predict(processed_data)[0]
    proba = model.predict_proba(processed_data)[0][prediction]
    
    # Create application record
    application = LoanApplication(
        applicant_name=applicant_name,
        gender=gender,
        married=married,
        dependents=dependents,
        education=education,
        self_employed=self_employed,
        applicant_income=applicant_income,
        coapplicant_income=coapplicant_income,
        loan_amount=loan_amount,
        loan_amount_term=loan_amount_term,
        credit_history=credit_history,
        property_area=property_area,
        prediction="Approved" if prediction == 1 else "Rejected",
        probability=proba
    )
    
    db.add(application)
    db.commit()
    db.refresh(application)
    
    # Load metrics
    metrics_data = db.query(ModelMetrics).first()
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "application": application,
        "metrics": metrics_data
    })

# Admin login form
@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_form(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request})

# Admin login
@app.post("/admin/login", response_class=RedirectResponse)
async def admin_login(
    request: Request,
    db: Session = Depends(get_db),
    username: str = Form(...),
    password: str = Form(...)
):
    user = authenticate_user(db, username, password)
    if not user or not user.is_admin:
        return templates.TemplateResponse("admin_login.html", {
            "request": request,
            "error": "Invalid credentials"
        })
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    response = RedirectResponse(url="/admin/dashboard", status_code=303)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

# Admin dashboard
@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not current_user or not current_user.is_admin:
        return RedirectResponse(url="/admin/login", status_code=303)
    
    applications = db.query(LoanApplication).order_by(LoanApplication.created_at.desc()).all()
    metrics_data = db.query(ModelMetrics).first()
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "applications": applications,
        "metrics": metrics_data,
        "current_user": current_user
    })

# Application detail
@app.get("/admin/application/{id}", response_class=HTMLResponse)
async def application_detail(
    request: Request,
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not current_user or not current_user.is_admin:
        return RedirectResponse(url="/admin/login", status_code=303)
    
    application = db.query(LoanApplication).filter(LoanApplication.id == id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    return templates.TemplateResponse("application_detail.html", {
        "request": request,
        "application": application
    })

# Update application status
@app.post("/admin/application/{id}/update", response_class=RedirectResponse)
async def update_application_status(
    id: int,
    status: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not current_user or not current_user.is_admin:
        return RedirectResponse(url="/admin/login", status_code=303)
    
    application = db.query(LoanApplication).filter(LoanApplication.id == id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    application.status = status
    db.commit()
    
    return RedirectResponse(url=f"/admin/application/{id}", status_code=303)

# Delete application
@app.get("/admin/application/{id}/delete", response_class=RedirectResponse)
async def delete_application(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not current_user or not current_user.is_admin:
        return RedirectResponse(url="/admin/login", status_code=303)
    
    application = db.query(LoanApplication).filter(LoanApplication.id == id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    db.delete(application)
    db.commit()
    
    return RedirectResponse(url="/admin/dashboard", status_code=303)

# Logout
@app.get("/admin/logout", response_class=RedirectResponse)
async def logout():
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie("access_token")
    return response

# New route for EDA report
@app.get("/eda-report", response_class=HTMLResponse)
async def get_eda_report():
    return FileResponse("artifacts/eda_report.html")

# Initialize admin user and metrics on startup
@app.on_event("startup")
async def startup_event():
    # Create tables
    create_tables()
    # Initialize admin user and metrics
    init_admin_user_and_metrics()