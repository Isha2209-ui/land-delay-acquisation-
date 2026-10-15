# Predictive Analytics System for Early Detection of Land Acquisition Delays

## 1. Project Overview

The **Predictive Analytics System for Early Detection of Land Acquisition Delays** is a machine learning project designed to estimate whether an infrastructure project is likely to experience delays.

The system uses historical project information to predict delay status, estimate delay probability, calculate a risk score, and categorize projects into risk levels. It also provides general recommendations to help users identify areas that may require further investigation.

**Implementation Details**

The primary implementation of this predictive analytics project is carried out in the Jupyter Notebook (`land_delay_analysis_fixed.ipynb`), which contains the main workflow for data analysis, preprocessing, model training, feature selection and evaluation.

The remaining Python scripts and supporting files are organised separately to perform individual tasks, such as data preparation, feature selection, model testing and prediction. The saved model (`land_delay_model.pkl`) is used to generate predictions without retraining the model each time. The terminal-based prediction script and web dashboard provide additional ways to use the trained model.

Thus, the Jupyter Notebook serves as the main development and analysis component, while the other files support the overall implementation.

## 2. Problem Statement

Infrastructure project delays can affect completion schedules, budgets, and resource planning. Identifying potentially delayed projects early can help authorities prioritize monitoring and investigate possible bottlenecks.

This project explores how machine learning can use historical infrastructure project data to support early delay-risk assessment.

## 3. Objectives

- Analyze historical infrastructure project data.
- Preprocess data for machine learning.
- Train a Logistic Regression classification model.
- Predict whether a project is likely to be delayed.
- Estimate the probability of delay.
- Assign a risk category based on the predicted probability.
- Display prediction results through a web dashboard.
- Evaluate model performance using standard classification metrics.

## 4. Dataset

The project uses the following dataset file:

`SIH26017_MoSPI_1477_projects.csv`

The dataset contains historical infrastructure project information used for analysis and prediction.

The model uses these input features:

- Sector
- Implementing agency
- State
- Original project cost
- Approval year
- Approval month
- Planned project duration

The target variable is `is_delayed`, which represents the project's delay status.

**Note:** Check the original dataset source and its redistribution conditions before making the dataset publicly available on GitHub.

## 5. Technology Stack

- **Programming language:** Python
- **Data processing:** Pandas and NumPy
- **Machine learning:** Scikit-learn
- **Model persistence:** Joblib
- **Model development:** Jupyter Notebook
- **Backend:** Python with FastAPI and Uvicorn
- **Frontend:** HTML, CSS, and JavaScript
- **Development environment:** Visual Studio Code

## 6. Machine Learning Methodology

The project follows these main steps:

1. Load and inspect the dataset.
2. Clean and preprocess the data.
3. Handle missing values.
4. Encode categorical features.
5. Standardize numerical features.
6. Select relevant features using `SelectKBest` and the ANOVA F-test.
7. Split the dataset into training, validation, and test sets.
8. Train the Logistic Regression classifier.
9. Evaluate the model using test data.
10. Save the trained model and preprocessing components for reuse.
11. Use the saved model to generate predictions through the application.

The notebook contains the data analysis, model training, and evaluation workflow.

## 7. Model Evaluation

The current notebook reports the following test-set results:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 85.14% |
| Precision | 89.36% |
| Recall | 87.50% |
| F1 Score | 0.8842 |
| ROC-AUC | 0.9389 |

### Meaning of the Metrics

- **Accuracy:** The proportion of all test predictions that were correct.
- **Precision:** The proportion of projects predicted as delayed that were actually delayed.
- **Recall:** The proportion of actually delayed projects correctly identified by the model.
- **F1 Score:** A combined measure of precision and recall.
- **ROC-AUC:** Measures how well the model distinguishes between the two outcome classes across different classification thresholds.

These values reflect the current notebook run. They should be updated if the model, dataset, or evaluation process changes.

## 8. Main Features

- Predict project delay status.
- Estimate the probability of project delay.
- Display a numerical risk score.
- Categorize delay risk as LOW, MEDIUM, HIGH, or CRITICAL.
- Provide general recommendations and investigation prompts.
- Display prediction results on a web dashboard.
- Explore project records through the dashboard.
- Run predictions through a terminal-based Python script.

The risk categories are based on probability thresholds defined in the application. Recommendations are general guidance, not confirmed explanations of why an individual project is delayed.

## 9. Project Structure

The main project files include:

- `backend/` — backend application code.
- `frontend/` — HTML, CSS, and JavaScript files for the dashboard.
- `land_delay_analysis.ipynb` — data analysis, model training, and evaluation.
- `SIH26017_MoSPI_1477_projects.csv` — dataset used by the project.
- `land_delay_model.pkl` — saved model and associated preprocessing components.
- `predict_project.py` — terminal-based prediction script.
- `risk_engine.py` — risk categorization and recommendation logic.
- `requirements.txt` — Python dependencies.
- `.gitignore` — files and folders excluded from version control.
- `README.md` — project documentation.

Other required files and data resources should be included according to the actual project configuration.

## 10. Installation and Setup

### Prerequisites

Install Python 3 and Visual Studio Code. Git is required if you are cloning the repository.

### Step 1: Clone the repository

Replace `YOUR_USERNAME` with your GitHub username and use the actual repository name.

```bash
git clone https://github.com/YOUR_USERNAME/land-delay-prediction.git
cd land-delay-prediction
```

Alternatively, open the existing project folder directly in VS Code.

### Step 2: Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

Make sure `requirements.txt` contains the libraries required by the actual application and notebook.

## 11. Running the Prediction Script

From the project root directory, run:

```bash
python predict_project.py
```

Follow the prompts in the terminal to enter project details and view the prediction.

The script requires the saved model and any other data files specified in its implementation.

## 12. Running the Web Dashboard

From the project root directory, activate the virtual environment and run:

```bash
python -m uvicorn backend.app:app --reload --port 8001
```

Once the server starts, open the following address in your browser:

**http://127.0.0.1:8001**

Keep the terminal running while using the dashboard.

To stop the server, press `Ctrl + C` in the terminal.

## 13. Running the Jupyter Notebook

Open `land_delay_analysis.ipynb` in Visual Studio Code.

Select the Python environment containing the required dependencies, then run the notebook cells in order.

The notebook can be used to inspect the dataset, reproduce the model development workflow, examine evaluation results, and understand how the model was trained.

The saved model allows the prediction application to reuse the trained components without retraining every time a prediction is requested.

## 14. Limitations

- Predictions depend on the quality and coverage of the available historical data.
- The model estimates the likelihood of delay; it cannot guarantee a future outcome.
- Model performance may change when applied to projects with different characteristics or data distributions.
- Risk categories depend on the thresholds defined in the application.
- General recommendations should not be treated as verified causes of delay.
- The system is an academic decision-support project and should not replace expert review or official project assessments.

## 15. Future Enhancements

Potential future improvements include:

- Comparing additional machine learning algorithms.
- Evaluating the model on newer project data.
- Improving feature engineering and model validation.
- Adding more detailed model explanations.
- Improving dashboard analytics and filtering.
- Integrating updated project data when reliable sources are available.

## 16. Conclusion

This project demonstrates how machine learning can be used to estimate infrastructure project delay risk from historical project information.

By combining data preprocessing, Logistic Regression, model evaluation, risk categorization, and a web dashboard, the system provides an interactive way to explore potential project delays and support further investigation.

## 17. Disclaimer

This project was developed for academic purposes. Its predictions are estimates based on the available data and should be reviewed by humans before being used to support any real-world decisions.
