# How to Use Prediction Numbers in Dry Eye Prediction

## 📋 Overview

The **Dry Eye Prediction** feature uses an MLP classifier to predict dry eye disease severity. After training, you can view predictions for individual test samples by entering a "prediction number" (which is an index into the test dataset).

---

## 🔢 What is a Prediction Number?

A **prediction number** is an **index** (0, 1, 2, 3, ...) that refers to a specific test sample in the dataset.

- **Range**: 0 to (Total Test Samples - 1)
- **Example**: If there are 100 test samples, valid numbers are 0 to 99

---

## 📊 How to Find Available Prediction Numbers

### **Step 1: Upload Your Dataset**
1. Go to **"Dry Eye Prediction"** in the menu
2. Upload a CSV or Excel file containing questionnaire data
3. The system will automatically:
   - Load the dataset
   - Split it into training (70%) and test (30%) sets
   - Train the MLP model
   - Generate predictions for all test samples

### **Step 2: Check the Output**
After training, you'll see:
- **Total no of input data**: Total records in your dataset
- **Total no of test data**: Number of test samples (30% of total)
- **Total no of train data**: Number of training samples (70% of total)

**Example:**
```
Total no of input data   : 200
Total no of test data    : 60
Total no of train data   : 140
```

In this case, **valid prediction numbers are 0 to 59**.

### **Step 3: View the Prediction Preview Table**
The system now displays a **preview table** showing:
- **Prediction #**: The index number (0, 1, 2, ...)
- **Predicted Class**: The severity stage (MILD STAGE, MODERATE STAGE, NORMAL, SEVERE STAGE)
- **Class Code**: The numeric code (0, 1, 2, or 3)

**Example Preview Table:**
| Prediction # | Predicted Class | Class Code |
|--------------|-----------------|------------|
| 0 | MILD STAGE | 0 |
| 1 | NORMAL | 2 |
| 2 | MODERATE STAGE | 1 |
| 3 | SEVERE STAGE | 3 |
| ... | ... | ... |

---

## 🎯 How to Enter a Prediction Number

### **Method 1: Use the Preview Table**
1. Look at the preview table displayed on the screen
2. Find a prediction number you're interested in
3. Enter that number in the input field
4. Click **"Submit"**

### **Method 2: Enter Any Valid Number**
1. Check the message: **"Valid Prediction Numbers: 0 to X"**
2. Enter any number within that range
3. Click **"Submit"**

---

## 📝 Step-by-Step Guide

### **Complete Workflow:**

1. **Upload Dataset**
   ```
   → Click "Dry Eye Prediction"
   → Upload CSV/Excel file
   → Wait for processing
   ```

2. **View Test Data Information**
   ```
   → See "Total no of test data: X"
   → Note: Valid numbers are 0 to (X-1)
   ```

3. **Check Preview Table**
   ```
   → Scroll to "Preview of Predictions"
   → See first 20 predictions with their classes
   → Note the prediction numbers you want to view
   ```

4. **Enter Prediction Number**
   ```
   → Type a number (e.g., "5")
   → Click "Submit"
   → See the result
   ```

5. **View Result**
   ```
   → System displays the predicted severity stage:
     - "Identified Affected = MILD STAGE"
     - "Identified Affected = MODERATE STAGE"
     - "Identified Normal"
     - "Identified Affected = SEVERE STAGE"
   ```

---

## 🎨 Prediction Class Codes

| Code | Severity Stage | Meaning |
|------|---------------|---------|
| **0** | MILD STAGE | Early stage dry eye disease |
| **1** | MODERATE STAGE | Moderate dry eye symptoms |
| **2** | NORMAL | No dry eye disease detected |
| **3** | SEVERE STAGE | Advanced dry eye disease |

---

## ⚠️ Common Issues and Solutions

### **Issue 1: "Invalid prediction number" Error**
**Problem**: You entered a number outside the valid range.

**Solution**: 
- Check the message showing valid range (e.g., "0 to 59")
- Enter a number within that range
- Remember: numbers start from 0, not 1

### **Issue 2: Don't Know Which Number to Enter**
**Problem**: You're not sure which prediction number to check.

**Solution**:
- Look at the preview table to see available predictions
- Start with 0 (first test sample)
- Try different numbers to see various predictions
- Each number represents a different patient/test case

### **Issue 3: "Please enter a valid number" Error**
**Problem**: You entered text or special characters instead of a number.

**Solution**:
- Enter only numeric digits (0, 1, 2, 3, ...)
- Don't include letters or special characters
- Don't include decimal points (use whole numbers only)

---

## 💡 Tips and Best Practices

1. **Start with 0**: The first test sample is always at index 0
2. **Use the Preview**: The preview table shows you what predictions are available
3. **Try Multiple Numbers**: Different test samples may have different predictions
4. **Check the Range**: Always verify the valid range before entering a number
5. **Understand the Context**: Each prediction number represents a different patient/test case from your dataset

---

## 🔍 Example Scenario

**Scenario**: You have a dataset with 150 records

1. **After Upload**:
   - Total input data: 150
   - Test data: 45 (30% of 150)
   - Train data: 105 (70% of 150)

2. **Valid Prediction Numbers**: 0 to 44

3. **Preview Table Shows**:
   - Prediction #0: MODERATE STAGE
   - Prediction #1: NORMAL
   - Prediction #2: MILD STAGE
   - ... (up to #19)

4. **To View Prediction #5**:
   - Enter "5" in the input field
   - Click "Submit"
   - See the result (e.g., "Identified Affected = MODERATE STAGE")

---

## 📌 Quick Reference

| What | How |
|------|-----|
| **Find valid range** | Check "Total no of test data" - valid numbers are 0 to (test_data - 1) |
| **See available predictions** | Look at the preview table |
| **Enter prediction number** | Type a number in the input field and click "Submit" |
| **View result** | Result appears below showing the severity stage |

---

## 🎓 Understanding the Test Split

The system automatically splits your data:
- **70%** → Training data (used to train the model)
- **30%** → Test data (used for predictions)

**Only test data predictions are available for viewing.**

If you want to predict on new data:
- Add new records to your dataset
- Re-upload the file
- The new records will be included in the test set

---

## ✅ Summary

1. **Prediction numbers are indices** (0, 1, 2, ...) into the test dataset
2. **Valid range**: 0 to (number of test samples - 1)
3. **Preview table** shows available predictions
4. **Enter any valid number** to see that test sample's prediction
5. **Result shows** the severity stage (MILD, MODERATE, NORMAL, or SEVERE)

---

**Need Help?** Check the preview table and error messages - they guide you to the correct prediction numbers!

