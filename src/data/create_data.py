import numpy as np
import pandas as pd


# Function to generate producers with capabilities
def generate_producers(n):
    producers = []
    for i in range(n):
        producers.append(
            {
                "producer_id": i + 1,
                "location_x": np.random.randint(
                    0, 101
                ),  # Location as integer between 0 and 100
                "location_y": np.random.randint(
                    0, 101
                ),  # Location as integer between 0 and 100
                "target_volume": np.random.randint(
                    30000, 60001
                ),  # Target volume as integer between 30k and 60k
                "current_volume": np.random.randint(
                    0, 50001
                ),  # Current volume as integer up to 50k
                "min_dimensions_x": np.random.randint(
                    1, 11
                ),  # Min dimensions as integers
                "min_dimensions_y": np.random.randint(1, 11),
                "max_dimensions_x": np.random.randint(
                    20, 51
                ),  # Max dimensions as integers
                "max_dimensions_y": np.random.randint(20, 51),
                "can_bend": np.random.choice([0, 1], p=[0.5, 0.5]),
                "can_deburr": np.random.choice([0, 1], p=[0.2, 0.8]),
                "can_galvanize": np.random.choice([0, 1], p=[0.1, 0.9]),
                "acceptance_rate": np.random.uniform(
                    0.3, 1.0
                ),  # Acceptance rate between 30% and 100%
            }
        )
    return pd.DataFrame(producers)


# Function to generate orders with dimensions and processing requirements
def generate_orders(n):
    orders = []
    for i in range(n):
        orders.append(
            {
                "order_id": i + 1,
                "client_location_x": np.random.randint(
                    0, 101
                ),  # Location as integer between 0 and 100
                "client_location_y": np.random.randint(
                    0, 101
                ),  # Location as integer between 0 and 100
                "order_value": round(
                    np.random.uniform(1000, 10000), 2
                ),  # Order value rounded to 2 decimals
                "dimensions_x": np.random.randint(
                    1, 51
                ),  # Dimensions as integers between 1 and 50
                "dimensions_y": np.random.randint(1, 51),
                "requires_bending": np.random.choice([0, 1], p=[0.7, 0.3]),
                "requires_deburring": np.random.choice([0, 1], p=[0.5, 0.5]),
                "requires_galvanizing": np.random.choice([0, 1], p=[0.3, 0.7]),
            }
        )
    return pd.DataFrame(orders)


if __name__ == "__main__":

    # Generate dataset´
    producers_df = generate_producers(20)
    orders_df = generate_orders(50)
    # Save to CSV (if needed)
    producers_df.to_csv("data/raw/producers.csv", index=False)
    orders_df.to_csv("data/raw/orders.csv", index=False)
