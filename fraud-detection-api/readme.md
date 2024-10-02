Certainly! Here's a README specifically for the `fraud-detection-api` directory, which contains the FastAPI application for your fraud detection project.

---

# 📦 Fraud Detection API

This directory contains the FastAPI application for serving the fraud detection model. The API is designed to receive transaction data and return predictions indicating whether a transaction is potentially fraudulent.

## 🚀 Running the API

### Prerequisites

- **Docker**: Ensure you have Docker installed. You can download it from [Docker's official website](https://www.docker.com/products/docker-desktop).

### Build and Run with Docker

1. **Navigate to the Project Root**:
   Make sure you are in the root directory of the project where the `docker-compose.yml` file is located.

2. **Build the Docker Image**:
   ```bash
   docker-compose build
   ```

3. **Start the API**:
   ```bash
   docker-compose up
   ```

   The API will be accessible at `http://localhost:8000`.

### Running Locally

If you prefer to run the API without Docker, ensure you have Python 3.8+ and the necessary dependencies installed.

1. **Install Dependencies**:
   Navigate to this directory and install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API**:
   Use Uvicorn to start the FastAPI application:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 🔥 API Endpoints

- **POST `/predict/`**: Receives transaction data and returns a prediction.

  - **Request Body**: JSON object with the following fields:

    ```json
    {
      "Time": 123549.0,
      "V1": 0.0324244087367978,
      "V2": 0.928688935050824,
      ...
      "V28": 0.077749433877637,
      "Amount": 17.99
    }
    ```

  - **Response**: JSON object containing the prediction result.

    ```json
    {
      "prediction": 0,
      "probability": 0.1234
    }
    ```

## ⚙️ Configuration

- **Environment Variables**:
  - `CACHE_PATH`: Directory path for caching data.

## 🛠️ Development

- **Adding New Features**: Add new endpoints or modify existing logic in `main.py`.
- **Testing**: Use tools like `pytest` to write and run tests for the API.

## 🤝 Contributions

Contributions are welcome! Feel free to submit issues or pull requests to enhance the functionality or fix bugs.

## 📜 License

This API is part of the Fraud Detection Project and is licensed under the MIT License.

## 📨 Contact

For any questions or inquiries, please contact Badrayour EL AMRI at [badrayour.elamri@protonmail.com](mailto:badrayour.elamri@protonmail.com).

---

Thank you for using the Fraud Detection API! 😊