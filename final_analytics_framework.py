"""
Comprehensive Analytical Framework for Vegetable Market Intelligence
=====================================================================
Updated version with actual dataset structure

Data Files:
- annex1.csv: Item catalog (Item Code, Item Name, Category Code, Category Name)
- annex2.csv: Sales transactions (Date, Time, Item Code, Quantity Sold, Unit Selling Price, Sale/Return, Discount)
- annex3.csv: Wholesale prices (Date, Item Code, Wholesale Price)
- annex4.csv: Loss rates (Item Code, Item Name, Loss Rate %)

Author: Data Analytics Team
Date: 2026-02-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DataIngestionPipeline:
    """
    Robust data ingestion pipeline with error handling and validation
    """
    
    def __init__(self, data_dir='dataset'):
        self.data_dir = data_dir
        self.raw_data = {}
        
    def load_data(self):
        """Load all CSV files with proper error handling"""
        try:
            # Load item catalog (annex1)
            self.raw_data['items'] = pd.read_csv(f'{self.data_dir}/annex1.csv')
            print(f"✓ Items catalog loaded: {len(self.raw_data['items'])} records")
            print(f"  Columns: {list(self.raw_data['items'].columns)}")
            
            # Load sales transactions (annex2) - if available
            annex2_path = f'{self.data_dir}/annex2.csv'
            if os.path.exists(annex2_path):
                self.raw_data['sales'] = pd.read_csv(annex2_path)
                print(f"✓ Sales transactions loaded: {len(self.raw_data['sales'])} records")
                print(f"  Columns: {list(self.raw_data['sales'].columns)}")
            else:
                print("⚠ Sales transactions (annex2.csv) not found - skipping")
            
            # Load wholesale prices (annex3)
            self.raw_data['prices'] = pd.read_csv(f'{self.data_dir}/annex3.csv')
            print(f"✓ Wholesale prices loaded: {len(self.raw_data['prices'])} records")
            print(f"  Columns: {list(self.raw_data['prices'].columns)}")
            
            # Load loss rates (annex4)
            self.raw_data['losses'] = pd.read_csv(f'{self.data_dir}/annex4.csv', encoding='utf-8-sig')
            print(f"✓ Loss rates loaded: {len(self.raw_data['losses'])} records")
            print(f"  Columns: {list(self.raw_data['losses'].columns)}")
            
            return self.raw_data
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise


class DataQualityChecker:
    """
    Handle data quality issues including missing values, outliers, and inconsistencies
    """
    
    @staticmethod
    def check_missing_values(df, df_name):
        """Identify missing values"""
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n{df_name} - Missing Values:")
            print(missing[missing > 0])
            return missing[missing > 0]
        else:
            print(f"✓ {df_name}: No missing values")
            return None
    
    @staticmethod
    def detect_outliers(series, method='iqr'):
        """Detect outliers using IQR or Z-score method"""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return pd.Series(dtype=bool)
            
        if method == 'iqr':
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(clean_series))
            outliers = pd.Series(False, index=series.index)
            outliers.loc[clean_series.index] = z_scores > 3
        return outliers
    
    @staticmethod
    def clean_numeric_column(series):
        """Clean numeric columns with proper type conversion"""
        if series.dtype == 'object':
            # Remove whitespace and convert
            series = series.str.strip()
            series = pd.to_numeric(series, errors='coerce')
        return series


class DataPreparationEngine:
    """
    Design optimal data structures for analytical workflows
    """
    
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.prepared_data = {}
        
    def prepare_master_dataset(self):
        """Create comprehensive master dataset with all features"""
        
        # Clean and prepare items data (annex1)
        items = self.raw_data['items'].copy()
        items.columns = items.columns.str.strip()
        
        # Clean and prepare wholesale prices (annex3)
        prices = self.raw_data['prices'].copy()
        prices.columns = prices.columns.str.strip()
        prices['Date'] = pd.to_datetime(prices['Date'])
        prices['Wholesale Price (RMB/kg)'] = DataQualityChecker.clean_numeric_column(
            prices['Wholesale Price (RMB/kg)']
        )
        
        # Clean and prepare loss rates (annex4)
        losses = self.raw_data['losses'].copy()
        losses.columns = losses.columns.str.strip()
        losses['Loss Rate (%)'] = DataQualityChecker.clean_numeric_column(
            losses['Loss Rate (%)']
        )
        
        # Merge prices with items
        master = prices.merge(items, on='Item Code', how='left')
        master = master.merge(losses[['Item Code', 'Loss Rate (%)']], on='Item Code', how='left')
        
        # Add temporal features
        master['Year'] = master['Date'].dt.year
        master['Month'] = master['Date'].dt.month
        master['Quarter'] = master['Date'].dt.quarter
        master['DayOfWeek'] = master['Date'].dt.dayofweek
        master['WeekOfYear'] = master['Date'].dt.isocalendar().week
        master['DayOfMonth'] = master['Date'].dt.day
        
        # Calculate effective price (accounting for loss)
        master['Effective Price (RMB/kg)'] = master['Wholesale Price (RMB/kg)'] * (
            1 + master['Loss Rate (%)'] / 100
        )
        
        # If sales data is available, merge it
        if 'sales' in self.raw_data:
            sales = self.raw_data['sales'].copy()
            sales.columns = sales.columns.str.strip()
            sales['Date'] = pd.to_datetime(sales['Date'])
            
            # Clean numeric columns
            numeric_cols = ['Quantity Sold (kilo)', 'Unit Selling Price (RMB/kg)']
            for col in numeric_cols:
                if col in sales.columns:
                    sales[col] = DataQualityChecker.clean_numeric_column(sales[col])
            
            # Calculate revenue
            if 'Quantity Sold (kilo)' in sales.columns and 'Unit Selling Price (RMB/kg)' in sales.columns:
                sales['Revenue (RMB)'] = sales['Quantity Sold (kilo)'] * sales['Unit Selling Price (RMB/kg)']
            
            # Aggregate sales by date and item
            sales_agg = sales.groupby(['Date', 'Item Code']).agg({
                'Quantity Sold (kilo)': 'sum',
                'Unit Selling Price (RMB/kg)': 'mean',
                'Revenue (RMB)': 'sum'
            }).reset_index()
            
            # Merge with master
            master = master.merge(sales_agg, on=['Date', 'Item Code'], how='left')
        
        self.prepared_data['master'] = master
        print(f"\n✓ Master dataset prepared: {len(master)} records, {len(master.columns)} features")
        print(f"  Date range: {master['Date'].min()} to {master['Date'].max()}")
        print(f"  Unique items: {master['Item Code'].nunique()}")
        print(f"  Categories: {master['Category Name'].nunique()}")
        
        return master


class ExploratoryDataAnalysis:
    """
    Identify statistically significant patterns, correlations, and anomalies
    """
    
    def __init__(self, master_data):
        self.data = master_data
        self.insights = {}
        
    def statistical_summary(self):
        """Generate comprehensive statistical summary"""
        print("\n" + "="*70)
        print("STATISTICAL SUMMARY")
        print("="*70)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        summary = self.data[numeric_cols].describe()
        print(summary)
        
        return summary
    
    def correlation_analysis(self):
        """Identify significant correlations"""
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS")
        print("="*70)
        
        # Select relevant numeric columns
        corr_cols = ['Wholesale Price (RMB/kg)', 'Loss Rate (%)', 'Effective Price (RMB/kg)']
        
        # Add sales columns if available
        if 'Quantity Sold (kilo)' in self.data.columns:
            corr_cols.append('Quantity Sold (kilo)')
        if 'Unit Selling Price (RMB/kg)' in self.data.columns:
            corr_cols.append('Unit Selling Price (RMB/kg)')
        if 'Revenue (RMB)' in self.data.columns:
            corr_cols.append('Revenue (RMB)')
        
        correlation_matrix = self.data[corr_cols].corr()
        
        print("\nCorrelation Matrix:")
        print(correlation_matrix.round(3))
        
        # Find strong correlations
        threshold = 0.5
        strong_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    strong_corr.append({
                        'Variable 1': correlation_matrix.columns[i],
                        'Variable 2': correlation_matrix.columns[j],
                        'Correlation': correlation_matrix.iloc[i, j]
                    })
        
        if strong_corr:
            print(f"\n✓ Strong correlations found (|r| > {threshold}):")
            for corr in strong_corr:
                print(f"  {corr['Variable 1']} <-> {corr['Variable 2']}: {corr['Correlation']:.3f}")
        
        self.insights['correlation'] = correlation_matrix
        return correlation_matrix
    
    def time_series_patterns(self):
        """Analyze temporal patterns and trends"""
        print("\n" + "="*70)
        print("TIME SERIES PATTERNS")
        print("="*70)
        
        # Monthly average prices
        monthly_avg = self.data.groupby(['Year', 'Month'])['Wholesale Price (RMB/kg)'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        print("\nMonthly Price Statistics (First 10 months):")
        print(monthly_avg.head(10))
        
        # Seasonal patterns
        seasonal_avg = self.data.groupby('Month')['Wholesale Price (RMB/kg)'].mean()
        print("\nSeasonal Average Prices:")
        print(seasonal_avg)
        
        # If sales data available, analyze sales trends
        if 'Quantity Sold (kilo)' in self.data.columns:
            monthly_sales = self.data.groupby(['Year', 'Month'])['Quantity Sold (kilo)'].sum().reset_index()
            print("\nMonthly Sales Volume (First 10 months):")
            print(monthly_sales.head(10))
            self.insights['monthly_sales'] = monthly_sales
        
        self.insights['monthly_trends'] = monthly_avg
        self.insights['seasonal_patterns'] = seasonal_avg
        
        return monthly_avg, seasonal_avg
    
    def category_analysis(self):
        """Analyze patterns by category"""
        print("\n" + "="*70)
        print("CATEGORY ANALYSIS")
        print("="*70)
        
        agg_dict = {
            'Wholesale Price (RMB/kg)': ['mean', 'std', 'min', 'max'],
            'Loss Rate (%)': 'mean',
            'Item Code': 'count'
        }
        
        # Add sales metrics if available
        if 'Quantity Sold (kilo)' in self.data.columns:
            agg_dict['Quantity Sold (kilo)'] = 'sum'
        if 'Revenue (RMB)' in self.data.columns:
            agg_dict['Revenue (RMB)'] = 'sum'
        
        category_stats = self.data.groupby('Category Name').agg(agg_dict).round(2)
        
        # Flatten column names
        category_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                  for col in category_stats.columns.values]
        category_stats = category_stats.sort_values('Wholesale Price (RMB/kg)_mean', ascending=False)
        
        print("\nCategory Performance:")
        print(category_stats)
        
        self.insights['category_stats'] = category_stats
        return category_stats
    
    def item_performance_analysis(self):
        """Analyze top and bottom performing items"""
        print("\n" + "="*70)
        print("ITEM PERFORMANCE ANALYSIS")
        print("="*70)
        
        # Average price by item
        item_stats = self.data.groupby(['Item Code', 'Item Name']).agg({
            'Wholesale Price (RMB/kg)': ['mean', 'std'],
            'Loss Rate (%)': 'mean',
        }).reset_index()
        
        item_stats.columns = ['Item Code', 'Item Name', 'Avg Price', 'Price Std', 'Avg Loss Rate']
        item_stats['Price CV'] = item_stats['Price Std'] / item_stats['Avg Price']
        
        # If sales data available
        if 'Revenue (RMB)' in self.data.columns:
            revenue_stats = self.data.groupby(['Item Code', 'Item Name'])['Revenue (RMB)'].sum().reset_index()
            item_stats = item_stats.merge(revenue_stats, on=['Item Code', 'Item Name'], how='left')
        
        print("\nTop 10 Items by Average Price:")
        print(item_stats.nlargest(10, 'Avg Price')[['Item Name', 'Avg Price', 'Avg Loss Rate']])
        
        print("\nTop 10 Items by Price Volatility:")
        print(item_stats.nlargest(10, 'Price CV')[['Item Name', 'Avg Price', 'Price CV']])
        
        print("\nTop 10 Items by Loss Rate:")
        print(item_stats.nlargest(10, 'Avg Loss Rate')[['Item Name', 'Avg Price', 'Avg Loss Rate']])
        
        self.insights['item_stats'] = item_stats
        return item_stats
    
    def anomaly_detection(self):
        """Detect anomalies and outliers"""
        print("\n" + "="*70)
        print("ANOMALY DETECTION")
        print("="*70)
        
        # Price outliers
        price_outliers = DataQualityChecker.detect_outliers(
            self.data['Wholesale Price (RMB/kg)'], method='iqr'
        )
        
        # Loss rate outliers
        loss_outliers = DataQualityChecker.detect_outliers(
            self.data['Loss Rate (%)'], method='iqr'
        )
        
        print(f"\n✓ Price outliers detected: {price_outliers.sum()} ({price_outliers.sum()/len(self.data)*100:.2f}%)")
        print(f"✓ Loss rate outliers detected: {loss_outliers.sum()} ({loss_outliers.sum()/len(self.data)*100:.2f}%)")
        
        # Get top anomalies
        anomalies = self.data[price_outliers | loss_outliers].copy()
        if len(anomalies) > 0:
            anomalies['Outlier Type'] = anomalies.apply(
                lambda x: 'Price' if price_outliers.get(x.name, False) else 'Loss Rate', axis=1
            )
            
            print("\nTop 10 Anomalies:")
            display_cols = ['Date', 'Item Name', 'Wholesale Price (RMB/kg)', 'Loss Rate (%)', 'Outlier Type']
            print(anomalies[display_cols].head(10))
        
        self.insights['anomalies'] = anomalies
        return anomalies
    
    def hypothesis_testing(self):
        """Generate and test hypotheses"""
        print("\n" + "="*70)
        print("HYPOTHESIS TESTING")
        print("="*70)
        
        # Hypothesis 1: Price differences across categories
        categories = self.data['Category Name'].unique()
        if len(categories) >= 2:
            cat1_prices = self.data[self.data['Category Name'] == categories[0]]['Wholesale Price (RMB/kg)'].dropna()
            cat2_prices = self.data[self.data['Category Name'] == categories[1]]['Wholesale Price (RMB/kg)'].dropna()
            
            if len(cat1_prices) > 0 and len(cat2_prices) > 0:
                t_stat, p_value = stats.ttest_ind(cat1_prices, cat2_prices)
                print(f"\nH1: Price difference between {categories[0]} and {categories[1]}")
                print(f"  t-statistic: {t_stat:.4f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")
        
        # Hypothesis 2: Correlation between loss rate and price
        valid_loss = self.data['Loss Rate (%)'].dropna()
        valid_price = self.data.loc[valid_loss.index, 'Wholesale Price (RMB/kg)'].dropna()
        valid_loss = valid_loss.loc[valid_price.index]
        
        if len(valid_loss) > 0 and len(valid_price) > 0:
            corr, p_value = stats.pearsonr(valid_loss, valid_price)
            print(f"\nH2: Correlation between Loss Rate and Price")
            print(f"  Correlation coefficient: {corr:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")


class VisualizationEngine:
    """
    Create interactive dashboards and visualizations
    """
    
    def __init__(self, master_data, insights):
        self.data = master_data
        self.insights = insights
        
    def create_comprehensive_dashboard(self, output_dir='outputs'):
        """Generate comprehensive visualization dashboard"""
        
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Price Distribution
        ax1 = plt.subplot(4, 3, 1)
        self.data['Wholesale Price (RMB/kg)'].hist(bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Wholesale Price (RMB/kg)', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title('Wholesale Price Distribution', fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # 2. Loss Rate Distribution
        ax2 = plt.subplot(4, 3, 2)
        self.data['Loss Rate (%)'].hist(bins=50, edgecolor='black', alpha=0.7, color='coral')
        plt.xlabel('Loss Rate (%)', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title('Loss Rate Distribution', fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # 3. Correlation Heatmap
        ax3 = plt.subplot(4, 3, 3)
        sns.heatmap(self.insights['correlation'], annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, linewidths=1, cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        
        # 4. Time Series - Average Price
        ax4 = plt.subplot(4, 3, 4)
        monthly_data = self.insights['monthly_trends']
        monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
        plt.plot(monthly_data['Date'], monthly_data['mean'], marker='o', linewidth=2, markersize=4)
        plt.fill_between(monthly_data['Date'], 
                        monthly_data['mean'] - monthly_data['std'],
                        monthly_data['mean'] + monthly_data['std'],
                        alpha=0.3)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Average Price (RMB/kg)', fontsize=10)
        plt.title('Price Trends Over Time', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 5. Seasonal Patterns
        ax5 = plt.subplot(4, 3, 5)
        seasonal = self.insights['seasonal_patterns']
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.bar(range(1, 13), seasonal.values, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Month', fontsize=10)
        plt.ylabel('Average Price (RMB/kg)', fontsize=10)
        plt.title('Seasonal Price Patterns', fontsize=12, fontweight='bold')
        plt.xticks(range(1, 13), months, fontsize=8)
        plt.grid(axis='y', alpha=0.3)
        
        # 6. Category Performance
        ax6 = plt.subplot(4, 3, 6)
        cat_stats = self.insights['category_stats']
        if 'Wholesale Price (RMB/kg)_mean' in cat_stats.columns:
            top_cats = cat_stats.nsmallest(10, 'Wholesale Price (RMB/kg)_mean')
            plt.barh(range(len(top_cats)), top_cats['Wholesale Price (RMB/kg)_mean'], color='teal', edgecolor='black')
            plt.yticks(range(len(top_cats)), top_cats.index, fontsize=8)
            plt.xlabel('Average Price (RMB/kg)', fontsize=10)
            plt.ylabel('Category', fontsize=10)
            plt.title('Categories by Average Price', fontsize=12, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
        
        # 7. Price vs Loss Rate Scatter
        ax7 = plt.subplot(4, 3, 7)
        scatter_data = self.data.sample(min(1000, len(self.data)))
        scatter = plt.scatter(scatter_data['Loss Rate (%)'], 
                   scatter_data['Wholesale Price (RMB/kg)'],
                   alpha=0.5, s=30, c=scatter_data['Month'], cmap='viridis')
        plt.colorbar(scatter, label='Month')
        plt.xlabel('Loss Rate (%)', fontsize=10)
        plt.ylabel('Wholesale Price (RMB/kg)', fontsize=10)
        plt.title('Price vs Loss Rate (colored by month)', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 8. Box Plot by Category (Top 5)
        ax8 = plt.subplot(4, 3, 8)
        top_categories = self.data['Category Name'].value_counts().head(5).index
        filtered_data = self.data[self.data['Category Name'].isin(top_categories)]
        filtered_data.boxplot(column='Wholesale Price (RMB/kg)', 
                            by='Category Name', ax=ax8)
        plt.xlabel('Category', fontsize=10)
        plt.ylabel('Wholesale Price (RMB/kg)', fontsize=10)
        plt.title('Price Distribution by Top Categories', fontsize=12, fontweight='bold')
        plt.suptitle('')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.grid(axis='y', alpha=0.3)
        
        # 9. Effective vs Wholesale Price
        ax9 = plt.subplot(4, 3, 9)
        sample_data = self.data.sample(min(500, len(self.data)))
        scatter = plt.scatter(sample_data['Wholesale Price (RMB/kg)'],
                   sample_data['Effective Price (RMB/kg)'],
                   alpha=0.6, c=sample_data['Loss Rate (%)'],
                   cmap='viridis', s=40)
        plt.colorbar(scatter, label='Loss Rate (%)')
        plt.xlabel('Wholesale Price (RMB/kg)', fontsize=10)
        plt.ylabel('Effective Price (RMB/kg)', fontsize=10)
        plt.title('Wholesale vs Effective Price', fontsize=12, fontweight='bold')
        max_price = sample_data['Wholesale Price (RMB/kg)'].max()
        plt.plot([0, max_price], [0, max_price], 'r--', linewidth=2, label='y=x', alpha=0.7)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 10. Top Items by Price
        ax10 = plt.subplot(4, 3, 10)
        if 'item_stats' in self.insights:
            top_items = self.insights['item_stats'].nlargest(10, 'Avg Price')
            plt.barh(range(len(top_items)), top_items['Avg Price'], color='darkgreen', edgecolor='black')
            plt.yticks(range(len(top_items)), top_items['Item Name'], fontsize=7)
            plt.xlabel('Average Price (RMB/kg)', fontsize=10)
            plt.title('Top 10 Most Expensive Items', fontsize=12, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
        
        # 11. Items by Loss Rate
        ax11 = plt.subplot(4, 3, 11)
        if 'item_stats' in self.insights:
            high_loss = self.insights['item_stats'].nlargest(10, 'Avg Loss Rate')
            plt.barh(range(len(high_loss)), high_loss['Avg Loss Rate'], color='crimson', edgecolor='black')
            plt.yticks(range(len(high_loss)), high_loss['Item Name'], fontsize=7)
            plt.xlabel('Average Loss Rate (%)', fontsize=10)
            plt.title('Top 10 Items by Loss Rate', fontsize=12, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
        
        # 12. Price Volatility
        ax12 = plt.subplot(4, 3, 12)
        if 'item_stats' in self.insights:
            volatile = self.insights['item_stats'].nlargest(10, 'Price CV')
            plt.barh(range(len(volatile)), volatile['Price CV'], color='orange', edgecolor='black')
            plt.yticks(range(len(volatile)), volatile['Item Name'], fontsize=7)
            plt.xlabel('Coefficient of Variation', fontsize=10)
            plt.title('Top 10 Most Volatile Items', fontsize=12, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Dashboard saved to {output_dir}/comprehensive_dashboard.png")
        plt.close()
        
        return fig


class StrategicInsightsGenerator:
    """
    Transform findings into business-relevant recommendations
    """
    
    def __init__(self, master_data, insights):
        self.data = master_data
        self.insights = insights
        
    def generate_executive_summary(self):
        """Create executive summary with key findings"""
        summary = {
            'data_overview': self._data_overview(),
            'key_findings': self._key_findings(),
            'recommendations': self._strategic_recommendations(),
            'risk_factors': self._identify_risks()
        }
        return summary
    
    def _data_overview(self):
        """Generate data overview"""
        overview = {
            'total_records': len(self.data),
            'date_range': f"{self.data['Date'].min()} to {self.data['Date'].max()}",
            'unique_items': self.data['Item Code'].nunique(),
            'categories': self.data['Category Name'].nunique(),
            'avg_price': self.data['Wholesale Price (RMB/kg)'].mean(),
            'avg_loss_rate': self.data['Loss Rate (%)'].mean()
        }
        
        if 'Revenue (RMB)' in self.data.columns:
            overview['total_revenue'] = self.data['Revenue (RMB)'].sum()
            overview['avg_daily_revenue'] = self.data.groupby('Date')['Revenue (RMB)'].sum().mean()
        
        return overview
    
    def _key_findings(self):
        """Extract key findings"""
        findings = []
        
        # Price trends
        monthly_trends = self.insights['monthly_trends']
        if len(monthly_trends) > 1:
            price_change = (monthly_trends['mean'].iloc[-1] - monthly_trends['mean'].iloc[0]) / monthly_trends['mean'].iloc[0] * 100
            findings.append(f"Overall price trend: {price_change:+.2f}% change over period")
        
        # Category insights
        cat_stats = self.insights['category_stats']
        if 'Wholesale Price (RMB/kg)_mean' in cat_stats.columns:
            highest_price_cat = cat_stats['Wholesale Price (RMB/kg)_mean'].idxmax()
            findings.append(f"Highest priced category: {highest_price_cat} (¥{cat_stats.loc[highest_price_cat, 'Wholesale Price (RMB/kg)_mean']:.2f}/kg)")
        
        if 'Loss Rate (%)_mean' in cat_stats.columns:
            highest_loss_cat = cat_stats['Loss Rate (%)_mean'].idxmax()
            findings.append(f"Highest loss rate category: {highest_loss_cat} ({cat_stats.loc[highest_loss_cat, 'Loss Rate (%)_mean']:.2f}%)")
        
        # Volatility
        if 'Wholesale Price (RMB/kg)_std' in cat_stats.columns and 'Wholesale Price (RMB/kg)_mean' in cat_stats.columns:
            cat_stats['Price Volatility'] = cat_stats['Wholesale Price (RMB/kg)_std'] / cat_stats['Wholesale Price (RMB/kg)_mean']
            most_volatile = cat_stats['Price Volatility'].idxmax()
            findings.append(f"Most volatile category: {most_volatile} (CV: {cat_stats.loc[most_volatile, 'Price Volatility']:.2f})")
        
        # Revenue insights (if available)
        if 'Revenue (RMB)_sum' in cat_stats.columns:
            top_revenue_cat = cat_stats['Revenue (RMB)_sum'].idxmax()
            findings.append(f"Highest revenue category: {top_revenue_cat} (¥{cat_stats.loc[top_revenue_cat, 'Revenue (RMB)_sum']:,.0f})")
        
        return findings
    
    def _strategic_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on loss rates
        high_loss_items = self.data[self.data['Loss Rate (%)'] > 15]
        if len(high_loss_items) > 0:
            recommendations.append({
                'priority': 'High',
                'area': 'Supply Chain Optimization',
                'recommendation': f'Focus on reducing waste for {high_loss_items["Item Name"].nunique()} items with loss rates >15%',
                'expected_impact': 'Potential 5-10% margin improvement',
                'implementation': 'Enhanced cold chain infrastructure, improved handling procedures, staff training'
            })
        
        # Based on price volatility
        cat_stats = self.insights['category_stats']
        if 'Wholesale Price (RMB/kg)_std' in cat_stats.columns and 'Wholesale Price (RMB/kg)_mean' in cat_stats.columns:
            cat_stats['CV'] = cat_stats['Wholesale Price (RMB/kg)_std'] / cat_stats['Wholesale Price (RMB/kg)_mean']
            volatile_cats = cat_stats[cat_stats['CV'] > 0.5]
            if len(volatile_cats) > 0:
                recommendations.append({
                    'priority': 'Medium',
                    'area': 'Pricing Strategy',
                    'recommendation': f'Implement dynamic pricing for {len(volatile_cats)} volatile categories',
                    'expected_impact': 'Better margin protection, reduced price risk',
                    'implementation': 'Real-time pricing algorithms, automated price adjustments'
                })
        
        # Based on seasonal patterns
        seasonal = self.insights['seasonal_patterns']
        peak_month = seasonal.idxmax()
        low_month = seasonal.idxmin()
        variation = (seasonal.max() - seasonal.min()) / seasonal.mean() * 100
        recommendations.append({
            'priority': 'Medium',
            'area': 'Inventory Management',
            'recommendation': f'Optimize inventory between peak (Month {peak_month}) and low (Month {low_month}) demand periods ({variation:.1f}% variation)',
            'expected_impact': 'Reduced carrying costs, improved cash flow',
            'implementation': 'Predictive inventory models, just-in-time procurement'
        })
        
        # Revenue-based recommendation (if sales data available)
        if 'Revenue (RMB)_sum' in cat_stats.columns:
            top_revenue = cat_stats['Revenue (RMB)_sum'].idxmax()
            recommendations.append({
                'priority': 'High',
                'area': 'Revenue Growth',
                'recommendation': f'Focus expansion efforts on top revenue category: {top_revenue}',
                'expected_impact': '15-20% revenue increase',
                'implementation': 'Expand product lines, increase marketing, optimize pricing'
            })
        
        return recommendations
    
    def _identify_risks(self):
        """Identify business risks"""
        risks = []
        
        # Price volatility risk
        price_std = self.data['Wholesale Price (RMB/kg)'].std()
        price_mean = self.data['Wholesale Price (RMB/kg)'].mean()
        cv = price_std / price_mean
        
        if cv > 0.5:
            risks.append({
                'risk': 'High Price Volatility',
                'severity': 'High',
                'description': f'Coefficient of variation: {cv:.2f}',
                'mitigation': 'Implement hedging strategies, negotiate long-term supplier contracts, diversify supplier base'
            })
        
        # Loss rate risk
        avg_loss = self.data['Loss Rate (%)'].mean()
        if avg_loss > 10:
            risks.append({
                'risk': 'High Average Loss Rate',
                'severity': 'Medium',
                'description': f'Average loss rate: {avg_loss:.2f}%',
                'mitigation': 'Invest in cold chain infrastructure, implement quality control procedures, staff training programs'
            })
        
        # Category concentration risk
        cat_stats = self.insights['category_stats']
        if 'Item Code_count' in cat_stats.columns:
            total_records = cat_stats['Item Code_count'].sum()
            top_cat_pct = cat_stats['Item Code_count'].max() / total_records * 100
            if top_cat_pct > 50:
                risks.append({
                    'risk': 'Category Concentration Risk',
                    'severity': 'Medium',
                    'description': f'Top category represents {top_cat_pct:.1f}% of records',
                    'mitigation': 'Diversify product portfolio, expand into new categories, develop alternative suppliers'
                })
        
        return risks
    
    def export_report(self, output_dir='outputs'):
        """Export comprehensive report"""
        summary = self.generate_executive_summary()
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("COMPREHENSIVE ANALYTICAL REPORT")
        report_lines.append("Vegetable Market Intelligence Analysis")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80)
        
        # Data Overview
        report_lines.append("\n" + "="*80)
        report_lines.append("1. DATA OVERVIEW")
        report_lines.append("="*80)
        for key, value in summary['data_overview'].items():
            report_lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        # Key Findings
        report_lines.append("\n" + "="*80)
        report_lines.append("2. KEY FINDINGS")
        report_lines.append("="*80)
        for i, finding in enumerate(summary['key_findings'], 1):
            report_lines.append(f"{i}. {finding}")
        
        # Strategic Recommendations
        report_lines.append("\n" + "="*80)
        report_lines.append("3. STRATEGIC RECOMMENDATIONS")
        report_lines.append("="*80)
        for i, rec in enumerate(summary['recommendations'], 1):
            report_lines.append(f"\nRecommendation {i}:")
            report_lines.append(f"  Priority: {rec['priority']}")
            report_lines.append(f"  Area: {rec['area']}")
            report_lines.append(f"  Action: {rec['recommendation']}")
            report_lines.append(f"  Impact: {rec['expected_impact']}")
            report_lines.append(f"  Implementation: {rec['implementation']}")
        
        # Risk Factors
        report_lines.append("\n" + "="*80)
        report_lines.append("4. RISK FACTORS")
        report_lines.append("="*80)
        for i, risk in enumerate(summary['risk_factors'], 1):
            report_lines.append(f"\nRisk {i}: {risk['risk']}")
            report_lines.append(f"  Severity: {risk['severity']}")
            report_lines.append(f"  Description: {risk['description']}")
            report_lines.append(f"  Mitigation: {risk['mitigation']}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        with open(f'{output_dir}/executive_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Executive report saved to {output_dir}/executive_report.txt")
        
        return report_text


def main():
    """
    Main execution pipeline
    """
    print("="*80)
    print("COMPREHENSIVE ANALYTICAL FRAMEWORK")
    print("Vegetable Market Intelligence Analysis")
    print("="*80)
    
    # 1. Data Ingestion
    print("\n[PHASE 1] DATA INGESTION & LOADING")
    print("-"*80)
    pipeline = DataIngestionPipeline()
    raw_data = pipeline.load_data()
    
    # 2. Data Quality Check
    print("\n[PHASE 2] DATA QUALITY ASSESSMENT")
    print("-"*80)
    for name, df in raw_data.items():
        DataQualityChecker.check_missing_values(df, name)
    
    # 3. Data Preparation
    print("\n[PHASE 3] DATA PREPARATION & ENGINEERING")
    print("-"*80)
    prep_engine = DataPreparationEngine(raw_data)
    master_data = prep_engine.prepare_master_dataset()
    
    # 4. Exploratory Data Analysis
    print("\n[PHASE 4] EXPLORATORY DATA ANALYSIS")
    print("-"*80)
    eda = ExploratoryDataAnalysis(master_data)
    eda.statistical_summary()
    eda.correlation_analysis()
    eda.time_series_patterns()
    eda.category_analysis()
    eda.item_performance_analysis()
    eda.anomaly_detection()
    eda.hypothesis_testing()
    
    # 5. Visualization
    print("\n[PHASE 5] VISUALIZATION & DASHBOARD CREATION")
    print("-"*80)
    viz_engine = VisualizationEngine(master_data, eda.insights)
    viz_engine.create_comprehensive_dashboard()
    
    # 6. Strategic Insights
    print("\n[PHASE 6] STRATEGIC INSIGHTS & RECOMMENDATIONS")
    print("-"*80)
    insights_gen = StrategicInsightsGenerator(master_data, eda.insights)
    report = insights_gen.export_report()
    print("\n" + report)
    
    # 7. Export processed data
    print("\n[PHASE 7] DATA EXPORT")
    print("-"*80)
    master_data.to_csv('outputs/master_dataset.csv', index=False)
    print("outputs/master_dataset.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nDeliverables:")
    print("  1. Comprehensive dashboard visualization (12 panels)")
    print("  2. Executive report with recommendations")
    print("  3. Master dataset with engineered features")
    print("  4. Reproducible code with clear documentation")
    print("\nAll outputs saved to: outputs/")


if __name__ == "__main__":
    main()