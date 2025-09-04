# Seedfolio Backend

This project is the backend for Seedfolio, a portfolio management application. It is built using Azure Functions and Python.

## Features

*   User registration and authentication using JWT.
*   Upload portfolio transactions from a CSV file.
*   View current portfolio holdings.
*   Get a summary of the portfolio performance.

## Technologies Used

*   [Azure Functions](https://azure.microsoft.com/en-us/products/functions/)
*   [Python](https://www.python.org/)
*   [SQLModel](https://sqlmodel.tiangolo.com/) for ORM
*   [pyodbc](https://github.com/mkleehammer/pyodbc) for SQL Server connection
*   [yfinance](https://pypi.org/project/yfinance/) for fetching stock data
*   [pandas](https://pandas.pydata.org/) for data manipulation
*   [passlib](https://passlib.readthedocs.io/en/stable/) for password hashing
*   [python-jose](https://python-jose.readthedocs.io/en/latest/) for JWT handling

## Getting Started

### Prerequisites

*   Python 3.10
*   An Azure account and the Azure Functions Core Tools.
*   A SQL Server instance.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/tamilselvan-eswaramoorthi/Seedfolio_backend.git
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up the environment variables by creating a `.env` file in the root directory. See the [Configuration](#configuration) section for more details.

4.  Start the Azure Functions host locally:
    ```bash
    func start
    ```

## API Endpoints

The following endpoints are available:

*   `POST /api/Register`: Register a new user.
*   `POST /api/Login`: Login a user and get a JWT token.
*   `GET /api/GetCurrentHoldings`: Get the current holdings for the authenticated user.
*   `GET /api/GetPortfolioSummary`: Get a summary of the portfolio for the authenticated user.
*   `POST /api/UploadTransactions`: Upload a CSV file with transactions for the authenticated user.

## Configuration

The application is configured using environment variables. Create a `.env` file in the root directory with the following variables:

```
DB_HOST=<your_db_host>
DB_PORT=<your_db_port>
DB_NAME=<your_db_name>
DB_USER=<your_db_user>
DB_PASSWORD=<your_db_password>
DB_SCHEMA=<your_db_schema>
JWT_SECRET=<your_jwt_secret>
JWT_ALGORITHM=<your_jwt_algorithm>
JWT_EXP_DELTA_SECONDS=<your_jwt_expiration_in_seconds>
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.


