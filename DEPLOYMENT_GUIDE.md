# Deployment Guide - Dry Eye Disease Detection System

## 🚀 Quick Deploy to Streamlit Cloud (Recommended)

### Step 1: Go to Streamlit Cloud
1. Visit: **https://share.streamlit.io/**
2. Sign in with your **GitHub account** (same account: Neelayaswanth)

### Step 2: Deploy Your App
1. Click **"New app"** button
2. Fill in the deployment form:
   - **Repository**: `Neelayaswanth/dry-eye-disease-detection`
   - **Branch**: `main`
   - **Main file path**: `Eye.py`
   - **App URL** (optional): Choose a custom subdomain or use auto-generated
3. Click **"Deploy"**

### Step 3: Wait for Deployment
- Streamlit Cloud will automatically:
  - Install all dependencies from `requirements.txt`
  - Build your application
  - Deploy it to a public URL
- This usually takes 2-5 minutes

### Step 4: Access Your App
- Once deployed, you'll get a URL like: `https://your-app-name.streamlit.app`
- Share this URL with users!

---

## 📋 Pre-Deployment Checklist

✅ **Completed:**
- [x] Repository pushed to GitHub
- [x] `requirements.txt` created
- [x] `.streamlit/config.toml` created
- [x] `Dataset.xlsx` included in repository
- [x] Large datasets excluded (Dataset/, Blink/)
- [x] Model files excluded (will be generated on first use)

---

## 🔧 Alternative Deployment Options

### Option 2: Heroku

1. **Install Heroku CLI**: https://devcenter.heroku.com/articles/heroku-cli

2. **Create Procfile**:
   ```
   web: streamlit run Eye.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Docker

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "Eye.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run**:
   ```bash
   docker build -t dry-eye-app .
   docker run -p 8501:8501 dry-eye-app
   ```

---

## ⚠️ Important Notes

### Files Not Included (By Design):
- **Dataset/** folder - Training images (too large)
- **Blink/** folder - Training images (too large)
- **Model files** (`.h5`) - Will be generated automatically on first use
- **Database files** (`.db`) - Will be created automatically

### First Run Behavior:
- The VGG-19 model will train automatically on first use (takes 3-5 minutes)
- After training, the model is saved and loads instantly on subsequent runs
- User database (`dbs.db`) is created automatically

### Required Files (Included):
- ✅ `Dataset.xlsx` - Required for Dry Eye Prediction feature
- ✅ All Python source files
- ✅ `requirements.txt`
- ✅ `.streamlit/config.toml`

---

## 🐛 Troubleshooting

### If deployment fails:

1. **Check requirements.txt**: Ensure all dependencies are listed
2. **Check file paths**: Make sure `Dataset.xlsx` is accessible
3. **Check logs**: Streamlit Cloud shows build logs in the dashboard
4. **Memory issues**: If model training fails, try reducing batch size in `Prediction.py`

### Common Issues:

**Issue**: "Module not found"
- **Solution**: Add missing package to `requirements.txt`

**Issue**: "File not found: Dataset.xlsx"
- **Solution**: Ensure file is committed to git (already done)

**Issue**: "Model training timeout"
- **Solution**: Model training happens on first use - be patient (3-5 minutes)

---

## 📊 Deployment Status

Your repository is ready for deployment:
- **GitHub**: https://github.com/Neelayaswanth/dry-eye-disease-detection
- **Status**: ✅ Ready to deploy
- **Main file**: `Eye.py`

---

## 🎉 After Deployment

Once deployed, your app will be accessible at:
- **Streamlit Cloud**: `https://[your-app-name].streamlit.app`

Users can:
1. Register new accounts
2. Login with credentials
3. Use all three prediction features:
   - Dry Eye Prediction (CSV/Excel upload)
   - Eye Disease Prediction (Image upload)
   - Eye Blink Detection (Image upload)

---

## 📞 Support

If you encounter any issues during deployment:
1. Check Streamlit Cloud logs
2. Verify all files are in the repository
3. Ensure `requirements.txt` is complete

