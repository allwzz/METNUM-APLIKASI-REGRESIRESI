import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    # Muat dataset dari file lokal
    df = pd.read_csv('student_performance.csv')
    return df

def plot_results_side_by_side(X, y, y_pred_linear, y_pred_other, xlabel, ylabel, title_linear, title_other):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot untuk Linear Regression
    axs[0].scatter(X, y, color='blue', label='Data Asli')
    axs[0].plot(X, y_pred_linear, color='red', label='Linear Regression')
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].set_title(title_linear)
    axs[0].legend()

    # Plot untuk Other Regression
    axs[1].scatter(X, y, color='blue', label='Data Asli')
    axs[1].plot(X, y_pred_other, color='green', label='Other Regression')
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    axs[1].set_title(title_other)
    axs[1].legend()

    plt.show()

def linear_regression(X, y):
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_pred = linear_model.predict(X)
    rms_error = np.sqrt(mean_squared_error(y, y_pred))
    return y_pred, rms_error

def polynomial_regression(X, y, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_pred = poly_model.predict(X_poly)
    rms_error = np.sqrt(mean_squared_error(y, y_pred))
    return y_pred, rms_error

def exponential_regression(X, y):
    y_log = np.log(y)
    exp_model = LinearRegression()
    exp_model.fit(X, y_log)
    y_pred_log = exp_model.predict(X)
    y_pred = np.exp(y_pred_log)
    rms_error = np.sqrt(mean_squared_error(y, y_pred))
    return y_pred, rms_error

def main():
    file_path = 'student_performance.csv' 
    df = load_data(file_path)

    nim = input("Masukkan NIM Anda: ")
    digit_terakhir_nim = int(nim[-1]) % 4

    if digit_terakhir_nim == 0:
        # Problem 1 dengan Metode 1 dan Metode 2
        X = df[['Hours Studied']].values
        y = df['Performance Index'].values

        print("Problem 1 dengan Metode 1 (Linear Regression):")
        y_pred_linear, rms_error_linear = linear_regression(X, y)
        print(f'RMS Error untuk Linear Model: {rms_error_linear}')

        print("Problem 1 dengan Metode 2 (Polynomial Regression):")
        y_pred_poly, rms_error_poly = polynomial_regression(X, y, 2)
        print(f'RMS Error untuk Polynomial Model: {rms_error_poly}')

        plot_results_side_by_side(X, y, y_pred_linear, y_pred_poly, 'Hours Studied', 'Performance Index', 
                                  'Linear Regression', 'Polynomial Regression')

    elif digit_terakhir_nim == 1:
        # Problem 1 dengan Metode 1 dan Metode 3
        X = df[['Hours Studied']].values
        y = df['Performance Index'].values

        print("Problem 1 dengan Metode 1 (Linear Regression):")
        y_pred_linear, rms_error_linear = linear_regression(X, y)
        print(f'RMS Error untuk Linear Model: {rms_error_linear}')

        print("Problem 1 dengan Metode 3 (Exponential Regression):")
        y_pred_exp, rms_error_exp = exponential_regression(X, y)
        print(f'RMS Error untuk Exponential Model: {rms_error_exp}')

        plot_results_side_by_side(X, y, y_pred_linear, y_pred_exp, 'Hours Studied', 'Performance Index', 
                                  'Linear Regression', 'Exponential Regression')

    elif digit_terakhir_nim == 2:
        # Problem 2 dengan Metode 1 dan Metode 2
        X = df[['Sample Question Papers Practiced']].values
        y = df['Performance Index'].values

        print("Problem 2 dengan Metode 1 (Linear Regression):")
        y_pred_linear, rms_error_linear = linear_regression(X, y)
        print(f'RMS Error untuk Linear Model: {rms_error_linear}')

        print("Problem 2 dengan Metode 2 (Polynomial Regression):")
        y_pred_poly, rms_error_poly = polynomial_regression(X, y, 2)
        print(f'RMS Error untuk Polynomial Model: {rms_error_poly}')

        plot_results_side_by_side(X, y, y_pred_linear, y_pred_poly, 'Sample Question Papers Practiced', 'Performance Index', 
                                  'Linear Regression', 'Polynomial Regression')

    elif digit_terakhir_nim == 3:
        # Problem 2 dengan Metode 1 dan Metode 3
        X = df[['Sample Question Papers Practiced']].values
        y = df['Performance Index'].values

        print("Problem 2 dengan Metode 1 (Linear Regression):")
        y_pred_linear, rms_error_linear = linear_regression(X, y)
        print(f'RMS Error untuk Linear Model: {rms_error_linear}')

        print("Problem 2 dengan Metode 3 (Exponential Regression):")
        y_pred_exp, rms_error_exp = exponential_regression(X, y)
        print(f'RMS Error untuk Exponential Model: {rms_error_exp}')

        plot_results_side_by_side(X, y, y_pred_linear, y_pred_exp, 'Sample Question Papers Practiced', 'Performance Index', 
                                  'Linear Regression', 'Exponential Regression')

if __name__ == "__main__":
    main()
