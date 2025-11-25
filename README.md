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
- **SciPy** — statistical functions (skewness, kurtosis, Z-score)
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

**Container Management:**

- `make build` - Build the Docker image
- `make up` - Start the containers
- `make down` - Stop the containers
- `make restart` - Restart the containers
- `make logs` - Show container logs
- `make shell` - Open a shell in the running container
- `make status` - Show container status

**Jupyter Notebook:**

- `make jupyter` - Start Jupyter Lab server (accessible at http://localhost:8888)
- `make logs-jupyter` - Show Jupyter container logs
- `make stop-jupyter` - Stop Jupyter container

**Cleanup:**

- `make clean` - Stop and remove containers, networks
- `make clean-containers` - Remove all stopped containers
- `make clean-images` - Remove unused Docker images
- `make clean-volumes` - Remove unused volumes
- `make clean-all` - Remove containers, images, and volumes (WARNING: destructive)

**Shortcuts:**

- `make rebuild` - Clean and rebuild the Docker image
- `make start` - Build and start the containers
- `make stop` - Stop the containers

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

## Dataset

The project uses enriched FIPE (Fundação Instituto de Pesquisas Econômicas) data for predicting used car prices in Brazil.

### Data Source

The raw data was downloaded from Kaggle:

- **Source**: [Average Car Prices Brazil](https://www.kaggle.com/datasets/vagnerbessa/average-car-prices-bazil?resource=download)
- **Author**: Vagner Bessa
- **Description**: Historical FIPE table data with average car prices in Brazil

### Data Enrichment

For educational purposes, the raw FIPE data has been enriched with synthetic but realistic values for missing features:

- **Quilometragem (km)**: Calculated based on vehicle age and price, following realistic usage patterns (average 15,000 km/year)
- **Location (state, city)**: Assigned based on Brazilian population distribution
- **Color**: Common car colors in Brazil (White, Silver, Black, etc.)
- **Doors**: Number of doors (2, 4, or 5)
- **Condition**: Vehicle condition (Regular, Good, Great, Excellent) based on age and mileage

The enrichment script (`src/data/enrich_fipe_data.py`) uses statistical patterns to generate realistic synthetic data that maintains the characteristics of the Brazilian used car market.

> **⚠️ Important**: These enriched features are synthetically generated for educational purposes. While they follow realistic patterns, they are not real-world observations. Model predictions should be validated against actual market data before any production use.

### Dataset Files

- **Raw data** (`data/raw/`):

  - `fipe_cars.csv` - Historical FIPE data (599,007 records)
  - `fipe_2022.csv` - 2022 FIPE data subset (290,275 records)

- **Processed data** (`data/processed/`):
  - `fipe_cars_enriched.csv` - Enriched historical data with all features
  - `fipe_2022_enriched.csv` - Enriched 2022 data with all features

### Data Schema

The enriched datasets include the following columns:

- `brand`, `model`, `year` - Vehicle identification
- `price` - Target variable (price in Brazilian Reais)
- `km` - Mileage in kilometers
- `state`, `city` - Location
- `fuel_type`, `transmission`, `engine_size` - Technical specifications
- `color`, `doors`, `condition` - Additional features
- `age_years` - Vehicle age

## Project Status

🚧 **In Development** - Current progress:

- ✅ Project structure and Docker configuration
- ✅ Data collection and enrichment (599k+ records)
- ✅ Exploratory Data Analysis (EDA) - Completed
  - ✅ Phase 1.1: Initial Data Exploration (`01_data_overview.ipynb`)
  - ✅ Phase 1.2: Target Variable Analysis (`02_target_analysis.ipynb`)
  - ✅ Phase 1.3: Feature Analysis (`03_feature_analysis.ipynb`)
  - ✅ Phase 1.4: Relationships and Correlations (`04_correlations.ipynb`)
  - ✅ Phase 1.5: Data Quality Assessment (`05_data_quality.ipynb`)
- 🚧 Data preprocessing and cleaning - Next
- ⏳ Feature engineering
- ⏳ Model development and training
- ⏳ API implementation
- ⏳ Model evaluation and deployment

**Development Strategy**: MVP-first approach - building essential features for a working end-to-end pipeline, then iterating with enhancements.

**Estimated Timeline**:

- MVP (Essential features): 10-12 weeks
- Full Implementation: 12-16 weeks

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed roadmap and task breakdown.

## License

See [LICENSE](LICENSE) file for details.
