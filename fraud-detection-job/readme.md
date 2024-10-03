# 📊 Fraud Detection Job

Welcome to the Fraud Detection Job! This project is designed to prepare, train, and evaluate a fraud detection model using machine learning techniques. The entire process is automated using Python scripts and Docker for easy deployment and execution.

## 🚀 Project Structure

The project consists of three main components:

1. **Data Preparation**: Download and prepare the dataset for training.
2. **Model Training**: Train a machine learning model to detect fraudulent activities.
3. **Model Evaluation**: Evaluate the model's performance using various metrics.

## 📂 Repository Structure

- `main.py`: The main entry point for executing the prepare, train, and evaluate commands.
- `tasks/`: Contains all tasks related to data preparation, training, and evaluation.
  - `tasks/prepare.py`: Prepares the dataset by downloading and processing it.
  - `tasks/train.py`: Trains the fraud detection model using the prepared dataset.
  - `tasks/evaluate.py`: Evaluates the trained model's performance.
- `docker-compose.yml`: Manages the Docker services for each task.

## 🛠️ Setup Instructions

### Prerequisites

- Docker
- Python 3.8+

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fraud-detection-job
   ```

2. **Build the Docker image**:
   ```bash
   docker-compose build
   ```

### Running the Project

Choose a model to train. Today two models are available and can be set using the environment variable `MODEL_NAME`. The values are `"linear_regression"` and `"xgboost_classifier"` 

1. **Prepare the Data**:
   ```bash
   docker-compose run fraud-detection-job-prepare
   ```

2. **Train the Model**:
   ```bash
   docker-compose run fraud-detection-job-train
   ```

3. **Evaluate the Model**:
   ```bash
   docker-compose run fraud-detection-job-evaluate
   ```

## 📝 Model Details

- **Algorithms**: [XGBoost Classifier, Linear Regressor]
- **Data Balancing**: SMOTE is used to balance the dataset during the preparation phase.
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1 Score.

## 📈 Output

- **Model**: Saved as `model.ubj` or `model.pkl` in the specified cache directory.
- **Evaluation Plot**: Log loss plot saved as `logloss.jpeg` to visualize model performance over epochs and Shapley values.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## 📜 License

This project is licensed under the MIT License.

## 📨 Contact

For any questions or inquiries, please contact Badrayour EL AMRI at [badrayour.elamri@protonmail.com](mailto:badrayour.elamri@protonmail.com).

---

Thank you for checking out the Fraud Detection Project! 😊