import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats


class KruskalWallisTest:
    def __init__(self, dataset, label_column='label'):
        """
        Initialize the KruskalWallisTest object.

        Args:
            dataset (pd.DataFrame): The dataset to perform the Kruskal-Wallis test on.
            label_column (str): The name of the column that contains the class labels (e.g., 'label').
        """
        self.dataset = dataset
        self.label_column = label_column
        self.phishing_data = dataset[dataset[label_column] == 1]  # Phishing URLs
        self.legitimate_data = dataset[dataset[label_column] == 0]  # Legitimate URLs
        self.feature_names = [col for col in dataset.columns if col != label_column]  # Features

    def perform_test(self):
        """
        Perform the Kruskal-Wallis test for each feature in the dataset.

        Returns:
            dict: A dictionary with feature names as keys and their Kruskal-Wallis test statistics as values.
        """
        Hs = {}

        for feature in self.feature_names:
            if self.phishing_data[feature].nunique() <= 1 or self.legitimate_data[feature].nunique() <= 1:
               print(f"Skipping {feature} because it has no variance in either class")
               continue
            # Perform the Kruskal-Wallis test on the feature
            st = stats.kruskal(self.phishing_data[feature].dropna(), self.legitimate_data[feature].dropna())
            Hs[feature] = st.statistic

        # Sort features based on the H-statistic (in descending order)
        Hs = sorted(Hs.items(), key=lambda x: x[1], reverse=True)

        return Hs

    def print_results(self, results):
        """
        Print the results of the Kruskal-Wallis test.

        Args:
            results (dict): The results of the Kruskal-Wallis test, sorted by statistic.
        """
        print("Ranked Features Based on Kruskal-Wallis Test:")
        for feature, statistic in results:
            print(f"{feature} --> {statistic}")


class FeaturePlotter:
    def __init__(self, dataset, kruskal_results, label_column='label'):
        """
        Initialize the FeaturePlotter object to visualize the features.

        Args:
            dataset (pd.DataFrame): The dataset to visualize the features.
            kruskal_results (list): List of tuples with feature names and Kruskal-Wallis statistics, sorted by significance.
            label_column (str): The name of the column that contains the class labels.
        """
        self.dataset = dataset
        self.kruskal_results = kruskal_results
        self.label_column = label_column

    def plot_features(self, top_n=5):
        """
        Plot the top N features using violin plots to compare their distributions in phishing vs legitimate URLs.

        Args:
            top_n (int): Number of top features to plot based on Kruskal-Wallis test significance.
        """
        top_features = [feature for feature, _ in self.kruskal_results[:top_n]]
        labels = ['Phishing', 'Legitimate']

        # Separate the dataset into phishing and legitimate classes
        phishing_data = self.dataset[self.dataset[self.label_column] == 1]
        legitimate_data = self.dataset[self.dataset[self.label_column] == 0]

        # Create a violin plot for each top feature
        for feature in top_features:
            fig = go.Figure()

            # Plot for phishing URLs
            fig.add_trace(go.Violin(
                y=phishing_data[feature],
                name='Phishing',
                box_visible=True,
                meanline_visible=True,
                points='all'
            ))

            # Plot for legitimate URLs
            fig.add_trace(go.Violin(
                y=legitimate_data[feature],
                name='Legitimate',
                box_visible=True,
                meanline_visible=True,
                points='all'
            ))

            # Update layout and display
            fig.update_layout(
                title=f"Distribution of {feature}",
                autosize=False,
                width=1200,
                height=600,
                font=dict(size=18, color="black")
            )

            fig.show()


import plotly.express as px
import numpy as np


class CorrelationMatrix:
    def __init__(self, dataset, kruskal_results, label_column='label'):
        """
        Initialize the CorrelationMatrix object to compute and plot correlations of top features.

        Args:
            dataset (pd.DataFrame): The dataset to calculate correlations for.
            kruskal_results (list): List of tuples with feature names and Kruskal-Wallis statistics, sorted by significance.
            label_column (str): The name of the column that contains the class labels.
        """
        self.dataset = dataset
        self.kruskal_results = kruskal_results
        self.label_column = label_column

    def plot_correlation_matrix(self, top_n=5):
        """
        Plot the correlation matrix of the top N features based on Kruskal-Wallis significance.

        Args:
            top_n (int): Number of top features to plot based on Kruskal-Wallis test significance.
        """
        top_features = [feature for feature, _ in self.kruskal_results[:top_n]]

        # Extract the data for the top features
        X = self.dataset[top_features].values

        # Calculate the correlation matrix
        corrMat = np.corrcoef(X, rowvar=False)

        # Plot the correlation matrix
        fig = px.imshow(corrMat,
                        text_auto=True,
                        labels=dict(x="Features", y="Features", color="Correlation"),
                        x=top_features,
                        y=top_features,
                        width=800,
                        height=800,
                        color_continuous_scale=px.colors.sequential.gray)

        fig.update_layout(title="Correlation Matrix of Top Features")
        fig.show()
