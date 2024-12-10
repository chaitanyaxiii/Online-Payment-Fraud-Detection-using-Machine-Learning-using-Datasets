import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import webbrowser

class FraudDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Online Payment Fraud Detection")
        self.root.geometry("800x800")
        
        self.df = None
        self.model = None
        self.cm_gb = None  # Confusion matrix for Gradient Boosting
        self.cm_logistic = None  # Confusion matrix for Logistic Regression
        
        # Indian Banks and Years
        self.banks = [
            "SBI", "Union Bank", "Punjab National Bank", "Bank of Baroda", "ICICI Bank",
            "Federal Bank", "HDFC Bank", "Axis Bank", "Kotak Mahindra Bank", "Bank of India"
        ]
        self.years = list(range(2015, 2025))
        
        self.create_widgets()

    def create_widgets(self):
        # Heading
        heading = tk.Label(self.root, text="Online Payment Fraud Detection", font=("Helvetica", 18, "bold"), bg="#4CAF50", fg="white")
        heading.pack(fill=tk.X)

        # Read Dataset Button
        read_button = tk.Button(self.root, text="Read Dataset", command=self.read_dataset, font=("Helvetica", 14), bg="#2196F3", fg="white")
        read_button.pack(pady=20)

        # Apply Algorithms
        algorithms_frame = tk.Frame(self.root)
        algorithms_frame.pack(pady=20)
        
        apply_gb_button = tk.Button(algorithms_frame, text="Apply HistGradientBoosting", command=self.apply_hist_gradient_boosting, font=("Helvetica", 14), bg="#FF9800", fg="white")
        apply_gb_button.grid(row=0, column=0, padx=10)
        
        apply_lr_button = tk.Button(algorithms_frame, text="Apply Logistic Regression", command=self.apply_logistic_regression, font=("Helvetica", 14), bg="#FF9800", fg="white")
        apply_lr_button.grid(row=0, column=1, padx=10)

        # Prediction Inputs
        self.inputs_frame = tk.Frame(self.root)
        self.inputs_frame.pack(pady=20)
        
        tk.Label(self.inputs_frame, text="Enter transaction details for prediction:", font=("Helvetica", 12, "bold")).grid(row=0, columnspan=2, pady=10)
        
        self.transaction_id_var = tk.StringVar()
        tk.Label(self.inputs_frame, text="Transaction ID:", font=("Helvetica", 12)).grid(row=1, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(self.inputs_frame, textvariable=self.transaction_id_var, font=("Helvetica", 12)).grid(row=1, column=1, padx=10, pady=5)
        
        self.bank_var = tk.StringVar()
        tk.Label(self.inputs_frame, text="Select Bank:", font=("Helvetica", 12)).grid(row=2, column=0, padx=10, pady=5, sticky="e")
        bank_dropdown = tk.OptionMenu(self.inputs_frame, self.bank_var, *self.banks)
        bank_dropdown.config(font=("Helvetica", 12))
        bank_dropdown.grid(row=2, column=1, padx=10, pady=5)
        self.bank_var.set(self.banks[0])  # Default selection
        
        self.year_var = tk.IntVar()
        tk.Label(self.inputs_frame, text="Select Year:", font=("Helvetica", 12)).grid(row=3, column=0, padx=10, pady=5, sticky="e")
        year_dropdown = tk.OptionMenu(self.inputs_frame, self.year_var, *self.years)
        year_dropdown.config(font=("Helvetica", 12))
        year_dropdown.grid(row=3, column=1, padx=10, pady=5)
        self.year_var.set(self.years[0])  # Default selection
        
        self.amount_var = tk.DoubleVar()
        tk.Label(self.inputs_frame, text="Amount:", font=("Helvetica", 12)).grid(row=4, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(self.inputs_frame, textvariable=self.amount_var, font=("Helvetica", 12)).grid(row=4, column=1, padx=10, pady=5)
        
        self.oldbalanceOrg_var = tk.DoubleVar()
        tk.Label(self.inputs_frame, text="Old Balance (Origin):", font=("Helvetica", 12)).grid(row=5, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(self.inputs_frame, textvariable=self.oldbalanceOrg_var, font=("Helvetica", 12)).grid(row=5, column=1, padx=10, pady=5)
        
        self.newbalanceOrig_var = tk.DoubleVar()
        tk.Label(self.inputs_frame, text="New Balance (Origin):", font=("Helvetica", 12)).grid(row=6, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(self.inputs_frame, textvariable=self.newbalanceOrig_var, font=("Helvetica", 12)).grid(row=6, column=1, padx=10, pady=5)
        
        self.oldbalanceDest_var = tk.DoubleVar()
        tk.Label(self.inputs_frame, text="Old Balance (Destination):", font=("Helvetica", 12)).grid(row=7, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(self.inputs_frame, textvariable=self.oldbalanceDest_var, font=("Helvetica", 12)).grid(row=7, column=1, padx=10, pady=5)
        
        self.newbalanceDest_var = tk.DoubleVar()
        tk.Label(self.inputs_frame, text="New Balance (Destination):", font=("Helvetica", 12)).grid(row=8, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(self.inputs_frame, textvariable=self.newbalanceDest_var, font=("Helvetica", 12)).grid(row=8, column=1, padx=10, pady=5)
        
        predict_button = tk.Button(self.inputs_frame, text="Predict Fraud", command=self.predict_fraud, font=("Helvetica", 14), bg="#9C27B0", fg="white")
        predict_button.grid(row=9, columnspan=2, pady=20)

        # Confusion Matrix and Graph Buttons
        matrix_button_lr = tk.Button(self.root, text="Go to Jupyter Notebook", command=self.go_to_jupyter_notebook, font=("Helvetica", 14), bg="#E91E63", fg="white")
        matrix_button_lr.pack(pady=20)
        
        graph_button_lr = tk.Button(self.root, text="Show Graph (LR)", command=self.show_logistic_graph, font=("Helvetica", 14), bg="#673AB7", fg="white")
        graph_button_lr.pack(pady=20)

        # Go to Jupyter Notebook Button
        jupyter_button = tk.Button(self.root, text="Go Notebook", command=self.go_to_jupyter_notebook, font=("Helvetica", 14), bg="#3F51B5", fg="white")
        jupyter_button.pack(pady=20)

    def read_dataset(self):
        self.df = pd.read_csv("onlinefraud.csv")  # Update with your dataset path
        messagebox.showinfo("Dataset", "Dataset read successfully!")
        print(self.df.head())
        
    def apply_hist_gradient_boosting(self):
        if self.df is not None:
            features = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
            X = self.df[features]
            y = self.df["isFraud"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            self.model = HistGradientBoostingClassifier(random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.cm_gb = confusion_matrix(y_test, y_pred)
            messagebox.showinfo("HistGradientBoosting", "HistGradientBoosting model applied successfully!")
        else:
            messagebox.showerror("Error", "Dataset not loaded")

    def apply_logistic_regression(self):
        if self.df is not None:
            features = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
            X = self.df[features]
            y = self.df["isFraud"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            self.model = LogisticRegression(random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.cm_logistic = confusion_matrix(y_test, y_pred)
            messagebox.showinfo("Logistic Regression", "Logistic Regression model applied successfully!")
        else:
            messagebox.showerror("Error", "Dataset not loaded")

    def predict_fraud(self):
        if self.model is not None:
            input_data = np.array([[self.amount_var.get(), self.oldbalanceOrg_var.get(), self.newbalanceOrig_var.get(), self.oldbalanceDest_var.get(), self.newbalanceDest_var.get()]])
            prediction = self.model.predict(input_data)
            result = "Fraud" if prediction[0] == 1 else "Not Fraud"
            bank_selected = self.bank_var.get()
            transaction_id = self.transaction_id_var.get()
            messagebox.showinfo("Prediction Result", f"The Transaction of the Given ID {transaction_id} is {result} of the bank {bank_selected}")
        else:
            messagebox.showerror("Error", "Model not trained")

    def show_gb_confusion_matrix(self):
        messagebox.showinfo("Info", "Gradient Boosting Confusion Matrix not available")

    def show_logistic_confusion_matrix(self):
        if self.cm_logistic is not None:
            fig, ax = plt.subplots()
            cm_display = ConfusionMatrixDisplay(self.cm_logistic, display_labels=['Not Fraud', 'Fraud']).plot(ax=ax, cmap=plt.cm.Blues)
            ax.set_title('Confusion Matrix (Logistic Regression)')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.root)
            canvas.get_tk_widget().pack()
            canvas.draw()
        else:
            messagebox.showerror("Error", "Logistic Regression Confusion Matrix not available")

    def show_gb_graph(self):
        messagebox.showinfo("Info", "Gradient Boosting Graph not available")

    def show_logistic_graph(self):
        x = np.linspace(-5, 5, 100)
        y = 1 / (1 + np.exp(-x))  # Logistic function
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability')
        ax.set_title('Logistic Regression Sigmoid Function')
        ax.grid(True)
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def go_to_jupyter_notebook(self):
        webbrowser.open("http://localhost:8888/notebooks/Online%20Payment%20Fraud%20Detection.ipynb")

if __name__ == "__main__":
    root = tk.Tk()
    app = FraudDetectionApp(root)
    root.mainloop()
