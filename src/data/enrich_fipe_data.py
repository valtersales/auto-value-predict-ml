"""
Script to enrich FIPE data with missing columns for the AutoValuePredict project.

This script adds synthetic but realistic values for:
- km (mileage) - based on vehicle age
- state, city - realistic Brazilian locations
- color, doors, condition - additional features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Optional


# Brazilian states with their major cities (weighted by population)
BRAZILIAN_STATES_CITIES = {
    'SP': ['São Paulo', 'Campinas', 'Guarulhos', 'São Bernardo do Campo', 'Santo André'],
    'RJ': ['Rio de Janeiro', 'Niterói', 'Duque de Caxias', 'Nova Iguaçu', 'São Gonçalo'],
    'MG': ['Belo Horizonte', 'Uberlândia', 'Contagem', 'Juiz de Fora', 'Betim'],
    'RS': ['Porto Alegre', 'Caxias do Sul', 'Pelotas', 'Canoas', 'Santa Maria'],
    'PR': ['Curitiba', 'Londrina', 'Maringá', 'Ponta Grossa', 'Cascavel'],
    'BA': ['Salvador', 'Feira de Santana', 'Vitória da Conquista', 'Camaçari', 'Juazeiro'],
    'SC': ['Florianópolis', 'Joinville', 'Blumenau', 'São José', 'Chapecó'],
    'GO': ['Goiânia', 'Aparecida de Goiânia', 'Anápolis', 'Rio Verde', 'Luziânia'],
    'PE': ['Recife', 'Jaboatão dos Guararapes', 'Olinda', 'Caruaru', 'Petrolina'],
    'CE': ['Fortaleza', 'Caucaia', 'Juazeiro do Norte', 'Maracanaú', 'Sobral'],
    'DF': ['Brasília'],
    'ES': ['Vitória', 'Vila Velha', 'Cariacica', 'Serra', 'Cachoeiro de Itapemirim'],
    'MT': ['Cuiabá', 'Várzea Grande', 'Rondonópolis', 'Sinop', 'Tangará da Serra'],
    'MS': ['Campo Grande', 'Dourados', 'Três Lagoas', 'Corumbá', 'Ponta Porã'],
    'PB': ['João Pessoa', 'Campina Grande', 'Santa Rita', 'Patos', 'Bayeux'],
    'AL': ['Maceió', 'Arapiraca', 'Rio Largo', 'Palmeira dos Índios', 'União dos Palmares'],
    'RN': ['Natal', 'Mossoró', 'Parnamirim', 'São Gonçalo do Amarante', 'Macaíba'],
    'PI': ['Teresina', 'Parnaíba', 'Picos', 'Piripiri', 'Floriano'],
    'MA': ['São Luís', 'Imperatriz', 'Caxias', 'Timon', 'Codó'],
    'SE': ['Aracaju', 'Nossa Senhora do Socorro', 'Lagarto', 'Itabaiana', 'São Cristóvão'],
    'TO': ['Palmas', 'Araguaína', 'Gurupi', 'Porto Nacional', 'Paraíso do Tocantins'],
    'AC': ['Rio Branco', 'Cruzeiro do Sul', 'Sena Madureira', 'Tarauacá', 'Feijó'],
    'AM': ['Manaus', 'Parintins', 'Itacoatiara', 'Manacapuru', 'Coari'],
    'RO': ['Porto Velho', 'Ji-Paraná', 'Ariquemes', 'Vilhena', 'Cacoal'],
    'RR': ['Boa Vista', 'Rorainópolis', 'Caracaraí', 'Alto Alegre', 'Mucajaí'],
    'AP': ['Macapá', 'Santana', 'Laranjal do Jari', 'Oiapoque', 'Mazagão'],
    'PA': ['Belém', 'Ananindeua', 'Marituba', 'Paragominas', 'Castanhal'],
}

# Common car colors in Brazil
CAR_COLORS = [
    'Branco', 'Prata', 'Preto', 'Cinza', 'Vermelho', 'Azul', 'Bege', 'Verde', 'Dourado', 'Marrom'
]

# Condition levels
CONDITIONS = ['Bom', 'Ótimo', 'Excelente', 'Regular']

# Door options
DOORS = [2, 4, 5]


def calculate_mileage(year_model: int, reference_year: int, price: float) -> float:
    """
    Calculate realistic mileage based on vehicle age and price.
    
    Assumes average of 15,000 km/year for regular use, with variation.
    Higher priced vehicles tend to have lower mileage.
    """
    age = reference_year - year_model
    
    if age <= 0:
        # New or very recent car
        base_km = np.random.uniform(0, 5000)
    elif age <= 1:
        base_km = np.random.uniform(5000, 20000)
    elif age <= 3:
        # 15k-20k km per year
        base_km = age * np.random.uniform(12000, 20000)
    elif age <= 5:
        # 15k-18k km per year
        base_km = age * np.random.uniform(13000, 18000)
    elif age <= 10:
        # 12k-16k km per year
        base_km = age * np.random.uniform(10000, 16000)
    else:
        # Older cars, 10k-14k km per year
        base_km = age * np.random.uniform(8000, 14000)
    
    # Adjust based on price (luxury cars tend to have lower mileage)
    if price > 100000:
        base_km *= np.random.uniform(0.7, 0.9)
    elif price > 50000:
        base_km *= np.random.uniform(0.85, 1.0)
    
    # Add some randomness
    km = base_km * np.random.uniform(0.9, 1.1)
    
    return max(0, round(km, 0))


def assign_location(seed: Optional[int] = None) -> tuple:
    """Assign a realistic Brazilian state and city."""
    if seed is not None:
        np.random.seed(seed)
    
    # Weight states by approximate population (SP, RJ, MG are more common)
    state_weights = {
        'SP': 0.22, 'RJ': 0.08, 'MG': 0.10, 'RS': 0.05, 'PR': 0.05,
        'BA': 0.07, 'SC': 0.03, 'GO': 0.03, 'PE': 0.04, 'CE': 0.04,
        'DF': 0.03, 'ES': 0.02, 'MT': 0.02, 'MS': 0.01, 'PB': 0.02,
        'AL': 0.01, 'RN': 0.01, 'PI': 0.01, 'MA': 0.03, 'SE': 0.01,
        'TO': 0.01, 'AC': 0.005, 'AM': 0.02, 'RO': 0.01, 'RR': 0.005,
        'AP': 0.005, 'PA': 0.02
    }
    
    states = list(state_weights.keys())
    weights = np.array(list(state_weights.values()))
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    state = np.random.choice(states, p=weights)
    city = np.random.choice(BRAZILIAN_STATES_CITIES[state])
    
    return state, city


def assign_condition(year_model: int, reference_year: int, km: float) -> str:
    """Assign condition based on age and mileage."""
    age = reference_year - year_model
    km_per_year = km / max(age, 1)
    
    if age <= 2 and km_per_year < 15000:
        return np.random.choice(['Ótimo', 'Excelente'], p=[0.3, 0.7])
    elif age <= 5 and km_per_year < 18000:
        return np.random.choice(['Bom', 'Ótimo'], p=[0.4, 0.6])
    elif age <= 10 and km_per_year < 20000:
        return np.random.choice(['Bom', 'Ótimo', 'Regular'], p=[0.5, 0.3, 0.2])
    else:
        return np.random.choice(['Bom', 'Regular'], p=[0.6, 0.4])


def enrich_fipe_data(
    input_file: str,
    output_file: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Enrich FIPE data with missing columns.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (if None, auto-generates)
        sample_size: If specified, only process this many rows
        random_seed: Random seed for reproducibility
        
    Returns:
        Enriched DataFrame
    """
    np.random.seed(random_seed)
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_seed)
        print(f"Sampling {len(df)} rows...")
    
    print(f"Original shape: {df.shape}")
    
    # Calculate age if not present
    if 'age_years' not in df.columns:
        df['age_years'] = df['year_of_reference'] - df['year_model']
    
    # Add mileage (km)
    print("Calculating mileage...")
    df['km'] = df.apply(
        lambda row: calculate_mileage(
            row['year_model'],
            row['year_of_reference'],
            row['avg_price_brl']
        ),
        axis=1
    )
    
    # Add location
    print("Assigning locations...")
    locations = [assign_location() for _ in range(len(df))]
    df['state'] = [loc[0] for loc in locations]
    df['city'] = [loc[1] for loc in locations]
    
    # Add color
    print("Assigning colors...")
    df['color'] = np.random.choice(CAR_COLORS, size=len(df), p=[0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03])
    
    # Add doors (most cars have 4 doors)
    print("Assigning door counts...")
    df['doors'] = np.random.choice(DOORS, size=len(df), p=[0.05, 0.85, 0.10])
    
    # Add condition
    print("Assigning conditions...")
    df['condition'] = df.apply(
        lambda row: assign_condition(
            row['year_model'],
            row['year_of_reference'],
            row['km']
        ),
        axis=1
    )
    
    # Rename columns to match project schema
    column_mapping = {
        'brand': 'brand',
        'model': 'model',
        'year_model': 'year',
        'avg_price_brl': 'price',
        'fuel': 'fuel_type',
        'gear': 'transmission',
        'engine_size': 'engine_size',
        'year_of_reference': 'year_of_reference',
        'month_of_reference': 'month_of_reference',
    }
    
    # Create new DataFrame with renamed columns
    enriched_df = pd.DataFrame()
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            enriched_df[new_col] = df[old_col]
    
    # Add new columns
    enriched_df['km'] = df['km']
    enriched_df['state'] = df['state']
    enriched_df['city'] = df['city']
    enriched_df['color'] = df['color']
    enriched_df['doors'] = df['doors']
    enriched_df['condition'] = df['condition']
    enriched_df['age_years'] = df['age_years']
    
    # Reorder columns to match expected schema
    column_order = [
        'brand', 'model', 'year', 'price', 'km', 'state', 'city',
        'fuel_type', 'transmission', 'engine_size', 'color', 'doors',
        'condition', 'age_years', 'year_of_reference', 'month_of_reference'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in enriched_df.columns]
    enriched_df = enriched_df[column_order]
    
    print(f"Enriched shape: {enriched_df.shape}")
    print(f"\nColumns: {list(enriched_df.columns)}")
    print(f"\nSample data:")
    print(enriched_df.head())
    print(f"\nData types:")
    print(enriched_df.dtypes)
    print(f"\nBasic statistics:")
    print(enriched_df[['price', 'km', 'year', 'age_years']].describe())
    
    # Save to file
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_enriched.csv"
    
    print(f"\nSaving enriched data to {output_file}...")
    enriched_df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(enriched_df)} rows to {output_file}")
    
    return enriched_df


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Enrich FIPE data with missing columns for AutoValuePredict"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input FIPE CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: input_file_enriched.csv)"
    )
    parser.add_argument(
        "-s", "--sample",
        type=int,
        default=None,
        help="Sample size (for testing, processes only N rows)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    enrich_fipe_data(
        args.input_file,
        args.output,
        args.sample,
        args.seed
    )


if __name__ == "__main__":
    main()

