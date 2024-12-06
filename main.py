import os
import subprocess
from random import random, uniform

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp


# Merge producer capabilities into a single capability set
def producer_capabilities(row):
    return set(
        [
            "bending" if row["can_bend"] else None,
            "deburring" if row["can_deburr"] else None,
            "galvanizing" if row["can_galvanize"] else None,
        ]
    ) - {None}


# Similarly, get order requirements
def order_requirements(row):
    return set(
        [
            "bending" if row["requires_bending"] else None,
            "deburring" if row["requires_deburring"] else None,
            "galvanizing" if row["requires_galvanizing"] else None,
        ]
    ) - {None}


# Simulate acceptance
def simulate_acceptance(row):
    return 1 if random() <= row["acceptance_rate"] else 0


if __name__ == "__main__":
    # Parameters
    num_iterations = 10  # Number of iterations to run the solver

    # Check if raw data exists
    data_folder = "data/raw/"
    producers_file = os.path.join(data_folder, "producers.csv")
    orders_file = os.path.join(data_folder, "orders.csv")

    if not (os.path.exists(producers_file) and os.path.exists(orders_file)):
        # Run data generation script
        print("Raw data not found. Generating data using create_data.py...")
        subprocess.run(["python", "src/data/create_data.py"])
    else:
        print("Raw data found. Proceeding with data loading...")

    # Load data
    producers_df = pd.read_csv(producers_file)
    orders_df = pd.read_csv(orders_file)

    # Initialize variables to track the best assignment
    best_total_net_benefit = float("-inf")
    best_final_assignments = None
    best_acceptance_rate = 0  # To track the acceptance rate of the best assignment

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")

        # Introduce variability by shuffling producers and orders
        producers_df_shuffled = producers_df.sample(frac=1).reset_index(drop=True)
        orders_df_shuffled = orders_df.sample(frac=1).reset_index(drop=True)

        # Slightly perturb acceptance rates to introduce randomness
        producers_df_shuffled["acceptance_rate_perturbed"] = producers_df_shuffled[
            "acceptance_rate"
        ] * np.random.uniform(0.95, 1.05, size=producers_df_shuffled.shape[0])
        producers_df_shuffled["acceptance_rate_perturbed"] = producers_df_shuffled[
            "acceptance_rate_perturbed"
        ].clip(
            0, 1
        )  # Ensure acceptance rates are between 0 and 1

        # Process capabilities and requirements
        producers_df_shuffled["capabilities"] = producers_df_shuffled.apply(
            producer_capabilities, axis=1
        )
        orders_df_shuffled["requirements"] = orders_df_shuffled.apply(
            order_requirements, axis=1
        )

        # Create compatibility matrix
        compatibility_matrix = pd.DataFrame(
            0,
            index=orders_df_shuffled["order_id"],
            columns=producers_df_shuffled["producer_id"],
        )

        # Calculate distances and build compatibility
        distance_matrix = pd.DataFrame(
            index=orders_df_shuffled["order_id"],
            columns=producers_df_shuffled["producer_id"],
            dtype=float,
        )

        for _, order in orders_df_shuffled.iterrows():
            for _, producer in producers_df_shuffled.iterrows():
                # Check capabilities
                if not order["requirements"].issubset(producer["capabilities"]):
                    continue
                # Check dimensions
                if (
                    producer["min_dimensions_x"]
                    <= order["dimensions_x"]
                    <= producer["max_dimensions_x"]
                    and producer["min_dimensions_y"]
                    <= order["dimensions_y"]
                    <= producer["max_dimensions_y"]
                ):
                    compatibility_matrix.at[
                        order["order_id"], producer["producer_id"]
                    ] = 1
                    # Calculate Euclidean distance
                    distance = np.sqrt(
                        (order["client_location_x"] - producer["location_x"]) ** 2
                        + (order["client_location_y"] - producer["location_y"]) ** 2
                    )
                    distance_matrix.at[order["order_id"], producer["producer_id"]] = (
                        distance
                    )

        # Remove incompatible pairs
        compatible_orders = compatibility_matrix.sum(axis=1) > 0
        compatibility_matrix = compatibility_matrix.loc[compatible_orders]
        distance_matrix = distance_matrix.loc[compatible_orders]

        # Calculate volume gaps
        producers_df_shuffled["volume_gap"] = (
            producers_df_shuffled["target_volume"]
            - producers_df_shuffled["current_volume"]
        )
        # Ensure no negative gaps
        producers_df_shuffled["volume_gap"] = producers_df_shuffled["volume_gap"].clip(
            lower=0
        )

        # Initialize the solver
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            print("Solver not found.")
            exit()

        # Decision variables: x[order_id][producer_id] = 1 if order is assigned to producer
        x = {}
        for order_id in compatibility_matrix.index:
            for producer_id in compatibility_matrix.columns:
                if compatibility_matrix.at[order_id, producer_id] == 1:
                    x[order_id, producer_id] = solver.BoolVar(
                        f"x_{order_id}_{producer_id}"
                    )

        # Objective: Maximize net benefit (expected accepted order value - transportation cost)
        objective = solver.Objective()
        for (order_id, producer_id), var in x.items():
            acceptance_rate = producers_df_shuffled.loc[
                producers_df_shuffled["producer_id"] == producer_id,
                "acceptance_rate_perturbed",
            ].values[0]
            order_value = orders_df_shuffled.loc[
                orders_df_shuffled["order_id"] == order_id, "order_value"
            ].values[0]
            expected_value = acceptance_rate * order_value

            # Get distance (transportation cost proxy)
            distance = distance_matrix.at[order_id, producer_id]
            transportation_cost = distance  # You can introduce a cost factor if needed

            # Net benefit
            net_benefit = expected_value - transportation_cost
            # Incorporate order value weighting (optional)
            net_benefit *= order_value / orders_df_shuffled["order_value"].max()

            # Objective coefficient
            objective.SetCoefficient(var, net_benefit)
        objective.SetMaximization()

        # Each order must be assigned to exactly one producer
        for order_id in compatibility_matrix.index:
            solver.Add(
                solver.Sum(
                    [
                        x[order_id, producer_id]
                        for producer_id in compatibility_matrix.columns
                        if (order_id, producer_id) in x
                    ]
                )
                == 1
            )

        # Solve the problem
        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            print("No optimal solution found.")
            continue

        # Extract assignments
        assignments = []
        for (order_id, producer_id), var in x.items():
            if var.solution_value() > 0.5:
                assignments.append({"order_id": order_id, "producer_id": producer_id})
        assignments_df = pd.DataFrame(assignments)

        # Merge assignments with data
        assignments_df = assignments_df.merge(
            producers_df_shuffled[
                ["producer_id", "acceptance_rate", "acceptance_rate_perturbed"]
            ],
            on="producer_id",
        )
        assignments_df = assignments_df.merge(
            orders_df_shuffled[
                ["order_id", "order_value", "client_location_x", "client_location_y"]
            ],
            on="order_id",
        )
        assignments_df = assignments_df.merge(
            producers_df_shuffled[
                ["producer_id", "location_x", "location_y", "volume_gap"]
            ],
            on="producer_id",
        )

        # Calculate distances and transportation costs
        assignments_df["distance"] = np.sqrt(
            (assignments_df["client_location_x"] - assignments_df["location_x"]) ** 2
            + (assignments_df["client_location_y"] - assignments_df["location_y"]) ** 2
        )
        assignments_df["transportation_cost"] = assignments_df[
            "distance"
        ]  # Adjust cost factor if needed

        # Simulate acceptance
        assignments_df["accepted"] = assignments_df.apply(simulate_acceptance, axis=1)

        # Handle rejections
        rejected_orders = assignments_df[assignments_df["accepted"] == 0][
            "order_id"
        ].tolist()

        # Reassignment loop
        while rejected_orders:
            # Remove producers who rejected from compatibility
            for order_id in rejected_orders:
                rejected_producer_id = assignments_df.loc[
                    assignments_df["order_id"] == order_id, "producer_id"
                ].values[0]
                compatibility_matrix.at[order_id, rejected_producer_id] = 0

            # Re-run the optimization
            # Initialize a new solver for reassignment
            solver = pywraplp.Solver.CreateSolver("SCIP")
            if not solver:
                print("Solver not found.")
                exit()

            # Decision variables for rejected orders
            x = {}
            for order_id in rejected_orders:
                for producer_id in compatibility_matrix.columns:
                    if compatibility_matrix.at[order_id, producer_id] == 1:
                        x[order_id, producer_id] = solver.BoolVar(
                            f"x_{order_id}_{producer_id}"
                        )

            # Objective: Maximize net benefit
            objective = solver.Objective()
            for (order_id, producer_id), var in x.items():
                acceptance_rate = producers_df_shuffled.loc[
                    producers_df_shuffled["producer_id"] == producer_id,
                    "acceptance_rate_perturbed",
                ].values[0]
                order_value = orders_df_shuffled.loc[
                    orders_df_shuffled["order_id"] == order_id, "order_value"
                ].values[0]
                expected_value = acceptance_rate * order_value

                # Get distance
                distance = distance_matrix.at[order_id, producer_id]
                transportation_cost = distance

                # Net benefit
                net_benefit = expected_value - transportation_cost
                net_benefit *= order_value / orders_df_shuffled["order_value"].max()

                objective.SetCoefficient(var, net_benefit)
            objective.SetMaximization()

            # Each rejected order must be assigned to exactly one producer
            for order_id in rejected_orders:
                solver.Add(
                    solver.Sum(
                        [
                            x[order_id, producer_id]
                            for producer_id in compatibility_matrix.columns
                            if (order_id, producer_id) in x
                        ]
                    )
                    == 1
                )

            # Solve the problem
            status = solver.Solve()

            if status != pywraplp.Solver.OPTIMAL:
                print("No optimal solution found during reassignment.")
                break

            # Extract new assignments
            new_assignments = []
            for (order_id, producer_id), var in x.items():
                if var.solution_value() > 0.5:
                    new_assignments.append(
                        {"order_id": order_id, "producer_id": producer_id}
                    )
            new_assignments_df = pd.DataFrame(new_assignments)

            # Merge with data
            new_assignments_df = new_assignments_df.merge(
                producers_df_shuffled[
                    ["producer_id", "acceptance_rate", "acceptance_rate_perturbed"]
                ],
                on="producer_id",
            )
            new_assignments_df = new_assignments_df.merge(
                orders_df_shuffled[
                    [
                        "order_id",
                        "order_value",
                        "client_location_x",
                        "client_location_y",
                    ]
                ],
                on="order_id",
            )
            new_assignments_df = new_assignments_df.merge(
                producers_df_shuffled[
                    ["producer_id", "location_x", "location_y", "volume_gap"]
                ],
                on="producer_id",
            )

            # Calculate distances and transportation costs
            new_assignments_df["distance"] = np.sqrt(
                (
                    new_assignments_df["client_location_x"]
                    - new_assignments_df["location_x"]
                )
                ** 2
                + (
                    new_assignments_df["client_location_y"]
                    - new_assignments_df["location_y"]
                )
                ** 2
            )
            new_assignments_df["transportation_cost"] = new_assignments_df["distance"]

            # Simulate acceptance again
            new_assignments_df["accepted"] = new_assignments_df.apply(
                simulate_acceptance, axis=1
            )

            # Update assignments
            for idx, row in new_assignments_df.iterrows():
                assignments_df.loc[
                    assignments_df["order_id"] == row["order_id"],
                    [
                        "producer_id",
                        "acceptance_rate",
                        "acceptance_rate_perturbed",
                        "order_value",
                        "client_location_x",
                        "client_location_y",
                        "location_x",
                        "location_y",
                        "volume_gap",
                        "distance",
                        "transportation_cost",
                        "accepted",
                    ],
                ] = row[
                    [
                        "producer_id",
                        "acceptance_rate",
                        "acceptance_rate_perturbed",
                        "order_value",
                        "client_location_x",
                        "client_location_y",
                        "location_x",
                        "location_y",
                        "volume_gap",
                        "distance",
                        "transportation_cost",
                        "accepted",
                    ]
                ]

            # Update rejected orders
            rejected_orders = new_assignments_df[new_assignments_df["accepted"] == 0][
                "order_id"
            ].tolist()

        # Final assignments
        final_assignments = assignments_df[assignments_df["accepted"] == 1]

        # Calculate total net benefit for this iteration
        total_net_benefit = (
            final_assignments["order_value"] * final_assignments["acceptance_rate"]
            - final_assignments["transportation_cost"]
        ).sum()

        # Calculate acceptance rate for this iteration
        total_orders = orders_df_shuffled["order_id"].nunique()
        accepted_orders = final_assignments["order_id"].nunique()
        acceptance_rate = accepted_orders / total_orders * 100

        print(f"Total Net Benefit: {total_net_benefit:.2f}")
        print(f"Acceptance Rate: {acceptance_rate:.2f}%")

        # Check if this is the best net benefit so far
        if total_net_benefit > best_total_net_benefit:
            best_total_net_benefit = total_net_benefit
            best_final_assignments = final_assignments.copy()
            best_acceptance_rate = acceptance_rate  # Store the acceptance rate
            print("New best total net benefit found.")

    # Output the best assignment
    print("\nBest Assignment Achieved:")
    print(best_final_assignments[["order_id", "producer_id"]])

    print(f"\nBest Total Net Benefit: {best_total_net_benefit:.2f}")
    print(f"Acceptance Rate: {best_acceptance_rate:.2f}%")

    # sort by producer_id and order_id and export
    best_final_assignments.sort_values(by=["producer_id", "order_id"], inplace=True)
    best_final_assignments.to_csv("data/processed/assigned_orders.csv", index=False)
    print("Best assignment exported to 'data/processed/assigned_orders.csv'.")
