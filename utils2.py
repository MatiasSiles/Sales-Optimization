# """
# toolkit for machine learning, deep learning, and advanced statistical analysis
# for business sales optimization

# Author: Matias Nicolas Siles
# Version: 1
# """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, KFold, TimeSeriesSplit
)

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
)

from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, mutual_info_regression,
    mutual_info_classif, RFE, RFECV
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor,
    GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
)

from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, silhouette_score
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# Statistical tests to check relationships and differences between data
from scipy import stats
from scipy.stats import (
    pearsonr, spearmanr, chi2_contingency, ttest_ind, mannwhitneyu,
    kruskal, shapiro, jarque_bera, anderson
)

class DataPreprocessor:
    """
    Comprehensive data preprocessing class for sales optimization analysis.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                      method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Args:
            X_train: Training features
            X_test: Test features (optional)
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            Scaled training and test features
        """
        numeric_cols = X_train.select_dtypes(include=["number"]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
        
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        
        self.scalers[method] = scaler
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None

class StatisticalAnalyzer:
    """
    Advanced statistical analysis class for sales data insights
    """
    
    def __init__(self):
        self.results = {}

    def descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive descriptive statistics analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing descriptive statistics
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        stats_summary = {
            'numeric_summary': df[numeric_cols].describe(),
            'categorical_summary': {},
            'missing_values': df.isnull().sum(),
            'data_types': df.dtypes,
            'shape': df.shape
        }
        
        # Categorical statistics
        for col in categorical_cols:
            stats_summary['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'value_counts': df[col].value_counts(),
                'mode': df[col].mode().iloc[0] if not df[col].mode().empty else None
            }
        
        # Distribution analysis
        stats_summary['skewness'] = df[numeric_cols].skew()
        stats_summary['kurtosis'] = df[numeric_cols].kurtosis()
        
        return stats_summary
    
    def correlation_analysis(self, df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix
        """
        numeric_df = df.select_dtypes(include=["number"])
        
        # i choose 3 tests, pearson for linear relationships and the other two for non-linear relationships
        if method == 'pearson':
            corr_matrix = numeric_df.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = numeric_df.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = numeric_df.corr(method='kendall')
        else:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
        
        return corr_matrix
    
    def hypothesis_testing(self, df: pd.DataFrame, group_col: str, 
                          target_col: str, test_type: str = 'auto') -> Dict[str, Any]:
        """
        Args:
            df: Input DataFrame
            group_col: Grouping column
            target_col: Target variable column
            test_type: Type of test ('auto', 'ttest', 'mann_whitney', 'kruskal')
            
        Returns:
            Test results dictionary
        """
        groups = df[group_col].unique() # the classes of the categorical variable are obtained
        
        if len(groups) == 2: # if the column has 2 or more categories
            group1 = df[df[group_col] == groups[0]][target_col].dropna() # We select the target data from a category
            group2 = df[df[group_col] == groups[1]][target_col].dropna()
            
            # I use both to know if the data is normal, Shapiro is for small data and Jarque-Bera is for large data
            _, p1 = shapiro(group1) if len(group1) <= 5000 else stats.jarque_bera(group1)[:2]
            _, p2 = shapiro(group2) if len(group2) <= 5000 else stats.jarque_bera(group2)[:2]
            
            if test_type == 'auto':
                if p1 > 0.05 and p2 > 0.05: # It is verified whether both groups have normal data (>0.05)
                    stat, p_value = ttest_ind(group1, group2)
                    test_used = 't-test' # assumes the data are normal, and compares the means and says if they are significant
                else:
                    stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                    test_used = 'Mann-Whitney U' # there is significance, Assumes that the data are not normal (has outliers), this test checks if there are very high or low numbers in each group.
            elif test_type == 'ttest':
                stat, p_value = ttest_ind(group1, group2)
                test_used = 't-test'
            elif test_type == 'mann_whitney':
                stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                test_used = 'Mann-Whitney U' 
                
        else:
            # Multiple groups
            group_data = [df[df[group_col] == group][target_col].dropna() for group in groups]
            stat, p_value = kruskal(*group_data)
            test_used = 'Kruskal-Wallis' # the same as Mann-Whitney U, but with several groups it detects some group with higher values
        
        return {
            'test_statistic': stat,
            'p_value': p_value,
            'test_used': test_used,
            'significant': p_value < 0.05,
            'groups': groups.tolist()
        }
    
    def time_series_analysis(self, df: pd.DataFrame, date_col: str, 
                           value_col: str, freq: str = 'D') -> Dict[str, Any]:
        """
        Args:
            df: Input DataFrame
            date_col: Date column name
            value_col: Value column name
            freq: Frequency for time series ('D', 'W', 'M')
            
        Returns:
            Time series analysis results
        """
        # Prepare time series data
        ts_df = df[[date_col, value_col]].copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col])
        ts_df = ts_df.groupby(date_col)[value_col].sum().resample(freq).sum() # We make 2 groupings, one to group the total sales by date and another to group the total sales by day, month or year
        
        # Seasonal decomposition
        if len(ts_df) >= 24:  # Minimum periods for decomposition(I will use it for months on the notebook, at least 24 months to decompose)
            decomposition = seasonal_decompose(ts_df, model='additive', period=12)
            
            results = {
                'original_series': ts_df,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'decomposition': decomposition
            }
        else:
            results = {
                'original_series': ts_df,
                'message': 'Insufficient data for decomposition, minimum 24 unique months required'
            }
        
        # Basic statistics
        results.update({
            'mean': ts_df.mean(),
            'std': ts_df.std(),
            'autocorrelation': ts_df.autocorr(lag=1) if len(ts_df) > 1 else None # correlation between the original data and the data compared by the previous one
        })
        
        return results
    

class CustomerSegmenter:

    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.rfm_thresholds = {}
        
    def rfm_segmentation(self, df):
        # Create a copy to avoid modifying original data
        rfm_df = df.copy()
        
        # We label customers' freshness days with 1-5 (5=many recency days, 1=few recency days)
        rfm_df['recency_score'] = pd.qcut(rfm_df['recency_days'], 
                                         q=5, labels=[5,4,3,2,1], duplicates='drop')
        rfm_df['frequency_score'] = pd.qcut(rfm_df['order_frequency'].rank(method='first'), 
                                           q=5, labels=[1,2,3,4,5], duplicates='drop')
        rfm_df['monetary_score'] = pd.qcut(rfm_df['total_spent'], 
                                          q=5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Convert to numeric
        rfm_df['recency_score'] = pd.to_numeric(rfm_df['recency_score'])
        rfm_df['frequency_score'] = pd.to_numeric(rfm_df['frequency_score'])
        rfm_df['monetary_score'] = pd.to_numeric(rfm_df['monetary_score'])
        
        # Calculate RFM combined score
        rfm_df['rfm_score'] = (rfm_df['recency_score'].astype(str) + 
                              rfm_df['frequency_score'].astype(str) + 
                              rfm_df['monetary_score'].astype(str))
        
        # Define segments based on RFM scores
        rfm_df['segments'] = rfm_df.apply(self._categorize_rfm, axis=1)
        
        return {
            'rfm_scores': rfm_df[['recency_score', 'frequency_score', 'monetary_score']],
            'rfm_combined': rfm_df['rfm_score'],
            'segments': rfm_df['segments']
        }
    
    def _categorize_rfm(self, row):
        r, f, m = row['recency_score'], row['frequency_score'], row['monetary_score']
        
        # Champions: High value, frequent, recent customers
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        
        # Loyal Customers: High frequency and monetary, but not necessarily recent
        elif f >= 4 and m >= 4:
            return 'Loyal Customers'
        
        # Potential Loyalists: Recent customers with good frequency/monetary
        elif r >= 4 and (f >= 3 or m >= 3):
            return 'Potential Loyalists'
        
        # New Customers: Very recent but low frequency/monetary
        elif r >= 4 and f <= 2 and m <= 2:
            return 'New Customers'
        
        # Promising: Recent customers with potential
        elif r >= 3 and f >= 2 and m >= 2:
            return 'Promising'
        
        # Need Attention: Above average recency, frequency, and monetary
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Need Attention'
        
        # About to Sleep: Below average recency, but decent frequency/monetary
        elif r <= 2 and f >= 2 and m >= 2:
            return 'About to Sleep'
        
        # At Risk: Good customers who haven't purchased recently
        elif r <= 2 and f >= 4 and m >= 4:
            return 'At Risk'
        
        # Cannot Lose Them: Very valuable customers who are inactive
        elif r <= 2 and f >= 4 and m >= 5:
            return 'Cannot Lose Them'
        
        # Hibernating: Low recency, frequency, and monetary
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating'
        
        # Lost: Lowest recency, frequency, and monetary
        else:
            return 'Lost'
    
    def kmeans_segmentation(self, data, n_clusters=5, random_state=42):
        # Prepare data
        X = data.fillna(0)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        # Get cluster centers (in original scale)
        cluster_centers = self.scaler.inverse_transform(self.kmeans_model.cluster_centers_)
        
        # Create cluster summary
        cluster_summary = self._create_cluster_summary(data, cluster_labels, cluster_centers)
        
        return {
            'labels': cluster_labels,
            'centers': cluster_centers,
            'silhouette_score': silhouette_avg,
            'inertia': self.kmeans_model.inertia_,
            'summary': cluster_summary
        }
    
    def find_optimal_clusters(self, data, max_clusters=10):
        X = data.fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        return {
            'k_values': list(K_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }

class MachineLearningOptimizer:
    """
    Comprehensive machine learning optimization class for sales prediction and analysis.
    """
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.feature_importance = {}
        self.results = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_regression_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        
        # Define regression models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'SVR': SVR(kernel='rbf'),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(random_state=42)
        
        
        results = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions, i will evalue the performance predictions on both train and test sets
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                results[name] = {
                    'model': model,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'predictions': test_pred
                }
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(
                        zip(X_train.columns, model.feature_importances_)
                    )
                
            except Exception as e:
                results[name] = {'error': str(e)}
        
        self.results['regression'] = results
        return results
