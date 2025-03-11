import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def model(x, params):
    """Nonlinear model: Exponential function."""
    a, b = params
    return a - a * np.exp(-b * x)

def jacobian(x, params):
    """Compute the Jacobian matrix (partial derivatives)."""
    a, b = params
    da = 1 - np.exp(-b * x)       # ∂y/∂a
    db = a * x * np.exp(-b * x)   # ∂y/∂b
    return np.column_stack((da, db))

def levenberg_marquardt_fit(file_path, num_iterations, lambda_):
    """Perform Levenberg-Marquardt optimization with manual iterations."""
    # Load the CSV file
    df = pd.read_csv(file_path)
    x_data = df['x'].values
    y_data = df['y'].values

    # Initial parameter guess: (a, b)
    params = np.array([40, 0.005])

    for iteration in range(1, num_iterations + 1):
        # Compute model predictions
        y_pred = model(x_data, params)
        
        # Compute residuals
        residuals = y_data - y_pred  # r = y - y_model
        
        # Compute Jacobian (partial derivatives)
        J = jacobian(x_data, params)

        # Compute J^T * J and J^T * residuals
        JTJ = J.T @ J
        JTr = J.T @ residuals

        # Add damping term (λ * I) to J^T * J
        JTJ_damped = JTJ + lambda_ * np.eye(JTJ.shape[0])

        # Solve for parameter updates Δp
        delta_params = np.linalg.solve(JTJ_damped, JTr)

        # Update parameters
        params += delta_params
        a, b = params  # Extract updated parameters

        # Compute additional columns
        dy_da = J[:, 0]
        dy_db = J[:, 1]
        
        dy_da_sq = dy_da ** 2
        dy_db_sq = dy_db ** 2
        dy_da_db = dy_da * dy_db
        res_dy_da = residuals * dy_da
        res_dy_db = residuals * dy_db
        residuals_sq = residuals ** 2  # (y - y_model)^2

        # Store iteration data in a DataFrame
        iteration_data = pd.DataFrame({
            "x": x_data,
            "y": y_data,
            "y_model": y_pred,
            "y - y_model": residuals,
            "(y - y_model)^2": residuals_sq,
            "dy/da": dy_da,
            "dy/db": dy_db,
            "(dy/da)^2": dy_da_sq,
            "(dy/db)^2": dy_db_sq,
            "(dy/da)*(dy/db)": dy_da_db,
            "(y-y_model)*(dy/da)": res_dy_da,
            "(y-y_model)*(dy/db)": res_dy_db,
            "delta_a" : delta_params[0],
            "Updated a": a,
            "delta_b" : delta_params[1],
            "Updated b": b
        })

        # Save iteration data to a CSV file
        filename = f"iteration_{iteration}.csv"
        iteration_data.to_csv(filename, index=False)
        print(f"Iteration {iteration} data saved to '{filename}' with updated a={a:.6f}, b={b:.6f}")

    # Final optimized parameters
    print("Final Optimized Parameters:", params)

    # Plot fitted curve
    plt.scatter(x_data, y_data, label="Data", color="red")
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = model(x_fit, params)
    plt.plot(x_fit, y_fit, label="Fitted Curve", color="blue")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Levenberg-Marquardt Nonlinear Least Squares Fit")
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = "data.csv"  # Change this to your actual CSV file
    lambda_ = 0.05
    num_iterations = 10
    levenberg_marquardt_fit(file_path, num_iterations, lambda_)  # Tune lambda_ as needed
