""" utils.py
This module contains utility functions for the main script.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from astroNN.nn.layers import MCDropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from scipy.stats import gaussian_kde

def import_data(property_path="df_properties.csv", 
                spectra_path="df_spectra.csv",
                carbon_name="C_tot",
                carbon_threshold=10,
                clay_name="ClyT_p",
                clay_threshold=40):
    """
    Import data from CSV files. The data is filtered by three criteria.
    1. The carbon content (column = carbon_name) is used to filter out organic soils with carbon content greater than 10%.
    2. Extreme values of clay content (column = clay_name) are removed by excluding the upper and lower quantiles (5% and 95%).
    3. The clay content is used to create two separate datasets:
        - One with clay content greater than 40% (in-domain)
        - Another with clay content less than 40% (out-of-domain)
    We rename the clay content column to "property" for consistency in future coding.
        
    Args:
        property_path (str): Path to the CSV file containing soil properties.
        spectra_path (str): Path to the CSV file containing spectral data.
        carbon_name (str): Column name for carbon content in the properties DataFrame.
        carbon_threshold (float): Threshold for carbon content to filter organic soils.
        clay_name (str): Column name for clay content in the properties DataFrame.
        clay_threshold (float): Threshold for clay content to separate datasets.
    
    Returns:
        in_domain_spec (np.ndarray): Spectral data for clay content greater than 40%.
        in_domain_prop (pd.DataFrame): Properties for clay content greater than 40%.
        X_independent (np.ndarray): Spectral data for clay content less than 40%.
        Y_independent (pd.DataFrame): Properties for clay content less than 40%.
    """
    df_sample = pd.read_csv(property_path)
    df_spectra = pd.read_csv(spectra_path)

    # 1. Filter out organic soils with carbon content greater than 10%
    df_sample1 = df_sample[df_sample[carbon_name] < carbon_threshold]
    logic = ~df_sample1[clay_name].isna()
    sample1 = df_sample1[["lay_id", clay_name]][logic]
    sample1 = sample1.rename(columns={clay_name: "property"})

    # 2. Remove extreme values by excluding the upper and lower quantiles (5% and 95%)
    upper_quantile = sample1["property"].quantile(0.95)
    lower_quantile = sample1["property"].quantile(0.05)
    filtered_data = sample1[(sample1["property"] > lower_quantile) & (sample1["property"] < upper_quantile)]

    # 3. Create two separate datasets based on clay content
    clay_40up = filtered_data[filtered_data["property"] > clay_threshold]
    clay_40main = filtered_data[filtered_data["property"] < clay_threshold]

    # Spectra data for clay > 40%
    lay_id_up = clay_40up["lay_id"]
    df_lay_id_up = pd.DataFrame({"lay_id": lay_id_up})
    df_merged_up = pd.merge(df_lay_id_up, df_spectra.iloc[:, 1:], on="lay_id", how="left")

    # Spectra data for clay < 40%
    lay_id_main = clay_40main["lay_id"]
    df_lay_id_main = pd.DataFrame({"lay_id": lay_id_main})
    df_merged_main = pd.merge(df_lay_id_main, df_spectra.iloc[:, 1:], on="lay_id", how="left")
    spec = np.expand_dims(df_merged_main.iloc[:, 1:].values, -1)

    in_domain_spec = spec
    in_domain_prop = clay_40main
    X_independent = np.expand_dims(df_merged_up.iloc[:, 1:].values, -1)
    Y_independent = clay_40up["property"]

    return in_domain_spec, in_domain_prop, X_independent, Y_independent

def reset():
    """
    Reset the TensorFlow session and clear the default graph.
    """
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

def create_model(train_data, lr=0.001, start_filters=32, dropout_rate=0.2):
    """
    Create a 1D Convolutional Neural Network model.
    
    Args:
        train_data (numpy.ndarray): The training data.
        lr (float): Learning rate for the optimizer.
        start_filters (int): Number of filters in the first convolutional layer.
        dropout_rate (float): Dropout rate for the dropout layers.

    Returns:
        model (tf.keras.Model): The compiled Keras model.
    """
    reset()
    
    input = Input(shape=train_data.shape[1:])

    x = Conv1D(start_filters, 5, activation="relu")(input)  # Convolutional layer 1
    x = MaxPool1D(2)(x)                                     # Max pooling layer 1   
    x = MCDropout(dropout_rate)(x)                          # Dropout layer 1
    
    x = Conv1D(start_filters * 2, 5, activation="relu")(x)  # Convolutional layer 2
    x = MaxPool1D(2)(x)                                     # Max pooling layer 2                                         
    x = MCDropout(dropout_rate)(x)                          # Dropout layer 2

    x = Conv1D(start_filters * 4, 5, activation="relu")(x)  # Convolutional layer 3
    x = MaxPool1D(2)(x)                                     # Max pooling layer 3
    x = MCDropout(dropout_rate)(x)                          # Dropout layer 3

    x = Conv1D(start_filters * 8, 5, activation="relu")(x)  # Convolutional layer 4
    x = MaxPool1D(2)(x)                                     # Max pooling layer 4
    x = MCDropout(dropout_rate)(x)                          # Dropout layer 4
    
    x = Flatten()(x) # Flatten

    output = Dense(1)(x)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=[])
    return model

def monte_carlo(model, test_X, scaler, num_predictions=100):
    """
    Perform forward passes of the model to generate a distribution of prediction.

    Args:
        model (tf.keras.Model): The trained Keras model.
        test_X (numpy.ndarray): The test data.
        scaler (MinMaxScaler): The scaler used to scale the data.
        num_predictions (int): Number of forward passes to perform.
    
    Returns:
        prediction_results (numpy.ndarray): Array of predictions from the model.
    """
    prediction_results = []

    for _ in range(num_predictions):
        Y_test_pred_scaled = model.predict(test_X, verbose=0)
        Y_test_pred = scaler.inverse_transform(Y_test_pred_scaled)
        Y_test_pred = Y_test_pred.reshape(-1)
        prediction_results.append(Y_test_pred)

    prediction_results = np.array(prediction_results)
    return prediction_results

def calculate_picp(prediction_results, Y_test, alpha=0.1, print_results=True):
    """
    Calculate the Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW).

    Args:
        prediction_results (numpy.ndarray): Array of predictions from the model.
        Y_test (numpy.ndarray): The true test labels.
        alpha (float): Significance level for the prediction interval.
        print_results (bool): Whether to print the results and plot the histogram.
    
    Returns:
        MPIW (float): Mean Prediction Interval Width.
        empirical_coverage (float): Empirical coverage probability.
    """
    # Calculate prediction interval as 95th and 5th quantile of the predictions
    n = len(Y_test)
    upper_bounds = []
    lower_bounds = []
    for i in range(0, n):
        upper_mc = np.quantile(prediction_results[:, i], np.ceil((n + 1) * (1-alpha/2)) / n, method="higher")
        lower_mc = np.quantile(prediction_results[:, i], np.ceil((n + 1) * (alpha/2)) / n, method="higher")
        upper_bounds.append(upper_mc)
        lower_bounds.append(lower_mc)

    # Convert lists to numpy arrays
    upper_bounds = np.array(upper_bounds)
    lower_bounds = np.array(lower_bounds)

    # Prediction interval for the test set
    prediction_sets = [lower_bounds, upper_bounds]
    PIW = upper_bounds - lower_bounds
    
    MPIW = np.mean(PIW)
    
    # Coverage
    empirical_coverage = ((Y_test >= prediction_sets[0]) & (Y_test <= prediction_sets[1])).mean()

    if print_results:
        print(f"The MPIW is {MPIW:.2f}")
        print(f"The PICP at {(1-alpha)*100}% is {empirical_coverage:.2f}")
        plt.hist(PIW, bins=50, edgecolor="k", density=True)
        plt.xlabel("Prediction Interval Width (PIW)")
        plt.ylabel("Density")
        plt.title("Density Distribution of Prediction Interval Width (PIW)")
        plt.show()

    return MPIW, empirical_coverage

def calculate_piw(prediction_results, Y_test, alpha=0.1):
    """
    Calculate the Prediction Interval Width (PIW) from the prediction results.

    Args:
        prediction_results (numpy.ndarray): Array of predictions from the model.
        Y_test (numpy.ndarray): The true test labels.
        alpha (float): Significance level for the prediction interval.
    
    Returns:
        PIW (numpy.ndarray): Prediction Interval Width for each test sample.
    """

    # Calculate prediction interval as 95th and 5th quantile of the predictions
    n = len(Y_test)
    upper_bounds = []
    lower_bounds = []
    for i in range(0, n):
        upper_mc = np.quantile(prediction_results[:, i], np.ceil((n + 1) * (1-alpha/2)) / n, method="higher")
        lower_mc = np.quantile(prediction_results[:, i], np.ceil((n + 1) * (alpha/2)) / n, method="higher")
        upper_bounds.append(upper_mc)
        lower_bounds.append(lower_mc)

    # Convert lists to numpy arrays
    upper_bounds = np.array(upper_bounds)
    lower_bounds = np.array(lower_bounds)

    # Prediction interval for the test set
    PIW = upper_bounds - lower_bounds

    return PIW

def conformal_prediction(model, calibration_X, calibration_Y, test_X, test_Y, scaler, alpha=0.1, print_results=False):
    """
    Calculate the Prediction Interval Width (PIW) and empirical coverage using conformal prediction.

    Args:
        model (tf.keras.Model): The trained Keras model.
        calibration_X (numpy.ndarray): The calibration input data.
        calibration_Y (numpy.ndarray): The calibration output data.
        test_X (numpy.ndarray): The test input data.
        test_Y (numpy.ndarray): The true test labels.
        scaler (MinMaxScaler): The scaler used to scale the data.
        alpha (float): Significance level for the prediction interval.
        print_results (bool): Whether to print the results.
    
    Returns:
        PIW (float): Prediction Interval Width.
        empirical_coverage (float): Empirical coverage probability.
    """
    
    prediction_sets = []
    empirical_coverage = []
    n = len(calibration_Y)
    # Calibration with external calibration set
    cal_predic = np.squeeze(np.asarray(model.predict(calibration_X, batch_size=1000, verbose=0)))
    cal_predic = scaler.inverse_transform(cal_predic.reshape(-1, 1)).flatten()
    cal_scores = np.abs(calibration_Y - cal_predic)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method="higher")
    PIW = qhat * 2
    if print_results:
        # print("The qhat is ", qhat)
        print(f"The PIW is {PIW:.2f} for every sample")
    
    # Prediction interval for the test set
    prediction = np.squeeze(np.asarray(model.predict(test_X, batch_size=1000, verbose=0)))
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
    prediction_sets = [prediction - qhat, prediction + qhat]

    # Coverage
    empirical_coverage = ((test_Y >= prediction_sets[0]) & (test_Y <= prediction_sets[1])).mean()
    if print_results:
        print(f"The PICP at {(1-alpha)*100}% is {empirical_coverage:.2f}")
    return PIW, empirical_coverage

def MC_CP_simple(calibration_Y, test_Y, mc_predictions_cali, mc_predictions_test, alpha=0.1, print_results=False):
    """
    Calculate the Prediction Interval Width (PIW) and empirical coverage using Monte Carlo Conformal Prediction.
    
    Args:
        calibration_Y (numpy.ndarray): The calibration output data.
        test_Y (numpy.ndarray): The true test labels.
        mc_predictions_cali (numpy.ndarray): Monte Carlo predictions for the calibration set.
        mc_predictions_test (numpy.ndarray): Monte Carlo predictions for the test set.
        alpha (float): Significance level for the prediction interval.
        print_results (bool): Whether to print the results and plot the histogram.
    
    Returns:
        MPIW (float): Mean Prediction Interval Width.
        empirical_coverage (float): Empirical coverage probability.
    """
    
    # Find 0.05 and 0.95 quantiles for calibration set MC predictions
    n = len(calibration_Y)
    upper_bounds_cali = []
    lower_bounds_cali = []
    for i in range(0, n):
        upper_mc = np.quantile(mc_predictions_cali[:, i], np.ceil((n + 1) * (1-alpha/2)) / n, method="higher")
        lower_mc = np.quantile(mc_predictions_cali[:, i], np.ceil((n + 1) * (alpha/2)) / n, method="higher")
        upper_bounds_cali.append(upper_mc)
        lower_bounds_cali.append(lower_mc)

    # Find 0.05 and 0.95 quantiles for test set MC predictions
    n = len(test_Y)
    upper_bounds_test = []
    lower_bounds_test = []
    for i in range(0, n):
        test_upper_mc = np.quantile(mc_predictions_test[:, i], np.ceil((n + 1) * (1-alpha/2)) / n, method="higher")
        test_lower_mc = np.quantile(mc_predictions_test[:, i], np.ceil((n + 1) * (alpha/2)) / n, method="higher")
        upper_bounds_test.append(test_upper_mc)
        lower_bounds_test.append(test_lower_mc)

    # Calculate conformity scores
    cal_scores = np.maximum(calibration_Y-upper_bounds_cali, lower_bounds_cali-calibration_Y)
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
    
    # Prediction interval for the test set
    prediction_sets = [lower_bounds_test - qhat, upper_bounds_test + qhat]
    PIW = prediction_sets[1] - prediction_sets[0]
    MPIW = np.mean(PIW)
    
    # Coverage (PICP)
    empirical_coverage = ((test_Y >= prediction_sets[0]) & (test_Y <= prediction_sets[1])).mean()

    # Print results
    if print_results:
        print(f"The qhat is {qhat:.2f}")
        print(f"The MPIW is {MPIW:.2f}")
        print(f"The PICP at {(1-alpha)*100}% is {empirical_coverage:.2f}")

        plt.hist(PIW, bins=50, edgecolor="k", density=True)
        plt.xlabel("Prediction Interval Width (PIW)")
        plt.ylabel("Density")
        plt.title("Density Distribution of Prediction Interval Width (PIW)")
        plt.show()   
    
    return MPIW, empirical_coverage

# Calculate PIW for MC-CP
def MC_CP_piw(calibration_Y, test_Y, mc_predictions_cali, mc_predictions_test, alpha=0.1):
    """
    Calculate the Prediction Interval Width (PIW) using Monte Carlo Conformal Prediction.

    Args:
        calibration_Y (numpy.ndarray): The calibration output data.
        test_Y (numpy.ndarray): The true test labels.
        mc_predictions_cali (numpy.ndarray): Monte Carlo predictions for the calibration set.
        mc_predictions_test (numpy.ndarray): Monte Carlo predictions for the test set.
        alpha (float): Significance level for the prediction interval.
    
    Returns:
        PIW (numpy.ndarray): Prediction Interval Width for each test sample.
    """
    # Find 0.05 and 0.95 quantiles for calibration set MC predictions
    n = len(calibration_Y)
    upper_bounds_cali = []
    lower_bounds_cali = []
    for i in range(0, n):
        upper_mc = np.quantile(mc_predictions_cali[:, i], np.ceil((n + 1) * (1-alpha/2)) / n, method="higher")
        lower_mc = np.quantile(mc_predictions_cali[:, i], np.ceil((n + 1) * (alpha/2)) / n, method="higher")
        upper_bounds_cali.append(upper_mc)
        lower_bounds_cali.append(lower_mc)

    # Find 0.05 and 0.95 quantiles for test set MC predictions
    n = len(test_Y)
    upper_bounds_test = []
    lower_bounds_test = []
    for i in range(0, n):
        test_upper_mc = np.quantile(mc_predictions_test[:, i], np.ceil((n + 1) * (1-alpha/2)) / n, method="higher")
        test_lower_mc = np.quantile(mc_predictions_test[:, i], np.ceil((n + 1) * (alpha/2)) / n, method="higher")
        upper_bounds_test.append(test_upper_mc)
        lower_bounds_test.append(test_lower_mc)

    # Calculate conformity scores
    cal_scores = np.maximum(calibration_Y-upper_bounds_cali, lower_bounds_cali-calibration_Y)
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
    
    # Prediction interval for the test set
    prediction_sets = [lower_bounds_test - qhat, upper_bounds_test + qhat]
    PIW = prediction_sets[1] - prediction_sets[0]
    return PIW
