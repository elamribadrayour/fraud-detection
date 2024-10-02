# 📊 Fraud Detection Project

Welcome to the Fraud Detection Project! This project is designed to detect fraudulent transactions using machine learning models. The application is built using FastAPI for serving predictions and Docker for containerization.

## 🚀 Project Structure

The project consists of several components:

- **fraud-detection-api**: Contains the FastAPI application for serving predictions.
- **fraud-detection-cache**: A directory used for caching data, such as model files and datasets.
- **fraud-detection-job**: Contains scripts related to data preparation, training, and evaluation of the model.

## 📁 Directory Overview

- `fraud-detection-api/`: This directory contains the source code for the FastAPI application.
- `fraud-detection-cache/`: This is where cached data and model files are stored.
- `fraud-detection-job/`: Contains Python scripts for preparing data, training the model, and evaluating the model.

## 🛠️ Setup Instructions

### Prerequisites

- **Docker**: Ensure you have Docker installed on your machine. You can download it from [Docker's official website](https://www.docker.com/products/docker-desktop).
- **Python 3.8+**: Necessary if you plan to run scripts locally outside of Docker.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd fraud-detection
   ```

2. **Build the Docker Image**:
   Navigate to your project directory and build the Docker image using the following command:
   ```bash
   docker-compose build
   ```

### Running the Project

- **Start the API**:
  Run the following command to start the FastAPI application in a Docker container:
  ```bash
  docker-compose up
  ```

  The API will be accessible at `http://localhost:8000`.

### API Usage

- **Endpoint**: `/predict/`
- **Method**: POST
- **Request Body**: JSON object with transaction features.

Example using `curl`:
```bash
curl -X POST "http://localhost:8000/predict/" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "Time": 123549.0,
  "V1": 0.0324244087367978,
  "V2": 0.928688935050824,
  "V3": -0.343407283021625,
  "V4": -0.620324374697907,
  "V5": 0.977567841434462,
  "V6": -0.491908261338232,
  "V7": 0.912720484797719,
  "V8": 0.027674175919869,
  "V9": -0.386914146755369,
  "V10": -0.696476846256358,
  "V11": 1.24226394882851,
  "V12": 0.866450273409973,
  "V13": 0.344650141053576,
  "V14": -0.89265842134895,
  "V15": -0.947357528057424,
  "V16": 0.415335814534708,
  "V17": 0.190179644156345,
  "V18": 0.103573595171275,
  "V19": -0.0632374102529764,
  "V20": 0.0917928089460491,
  "V21": -0.258784981151548,
  "V22": -0.643562188406274,
  "V23": 0.142541234317062,
  "V24": 0.611408903485703,
  "V25": -0.459356058816613,
  "V26": 0.0903803875417543,
  "V27": 0.218078472698999,
  "V28": 0.077749433877637,
  "Amount": 17.99
}'
```

### 📈 Model Details

- **Algorithm**: XGBoost Classifier
- **Features**: Includes 28 anonymized features (V1-V28), `Time`, and `Amount`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## 📜 License

This project is licensed under the MIT License.

## 📨 Contact

For any questions or inquiries, please contact Badrayour EL AMRI at [badrayour.elamri@protonmail.com](mailto:badrayour.elamri@protonmail.com).

---

Thank you for checking out the Fraud Detection Project! 😊