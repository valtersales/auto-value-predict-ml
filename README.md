# AutoValuePredict ML

AutoValuePredict is a machine learning project designed to predict the market value of used cars in Brazil. It demonstrates end-to-end ML development, including data collection, cleaning, feature engineering, regression modeling, model evaluation, and deployment.

This project aims to serve as a practical and professional showcase of data science and machine learning skills, while also producing a reusable prediction pipeline and a simple API for real-time price estimation.

## Project Structure

```
auto-value-predict-ml/
├── .dockerignore          # Docker ignore patterns
├── .gitignore            # Git ignore patterns
├── Dockerfile            # Docker image configuration
├── docker-compose.yml    # Docker Compose configuration
├── Makefile              # Convenient commands for Docker operations
├── pyproject.toml        # Poetry dependency management
├── requirements.txt      # pip dependency management (alternative)
├── data/
│   ├── raw/             # Raw data files
│   └── processed/       # Processed/cleaned data files
├── src/
│   ├── data/            # Data collection and loading modules
│   ├── features/        # Feature engineering modules
│   ├── models/          # Model training and evaluation modules
│   └── api/             # FastAPI application for serving predictions
├── notebooks/           # Jupyter notebooks for exploratory analysis
├── models/              # Trained model artifacts
└── tests/               # Unit and integration tests
```

## Technologies

### Languages & Core
- **Python 3.10+**

### Data & ML
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **scikit-learn** — ML algorithms, preprocessing, pipelines
- **Matplotlib / Seaborn** — data visualization (exploratory analysis)
- **XGBoost / LightGBM** — boosted models for high accuracy

### API & Deployment
- **FastAPI** — REST API for serving predictions
- **Uvicorn** — application server
- **Docker** — containerization for portability
- **Poetry / pip** — dependency management

## Getting Started

### Prerequisites
- Docker and Docker Compose installed
- (Optional) Python 3.10+ and Poetry/pip for local development

### Using Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   make build
   ```

2. **Start the containers:**
   ```bash
   make up
   ```

3. **View logs:**
   ```bash
   make logs
   ```

4. **Stop the containers:**
   ```bash
   make down
   ```

### Available Make Commands

Run `make help` to see all available commands:

- `make build` - Build the Docker image
- `make up` - Start the containers
- `make down` - Stop the containers
- `make restart` - Restart the containers
- `make logs` - Show container logs
- `make shell` - Open a shell in the running container
- `make clean` - Stop and remove containers, networks
- `make clean-containers` - Remove all stopped containers
- `make clean-images` - Remove unused Docker images
- `make clean-volumes` - Remove unused volumes
- `make clean-all` - Remove containers, images, and volumes (WARNING: destructive)
- `make rebuild` - Clean and rebuild the Docker image
- `make start` - Build and start the containers
- `make stop` - Stop the containers
- `make status` - Show container status

### Local Development

If you prefer to run locally without Docker:

1. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

## Project Status

🚧 **In Development** - Project structure and Docker configuration are set up. Next steps include:
- Data collection and preprocessing
- Feature engineering
- Model development and training
- API implementation
- Model evaluation and deployment

## License

See [LICENSE](LICENSE) file for details.
