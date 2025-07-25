import pandas as pd
import numpy as np

def analyze_sales_data():
    """Analyze sample sales data"""
    print("📊 Starting Data Analysis...")
    
    # Create sample data
    data = {
        'product': ['A', 'B', 'C', 'A', 'B', 'C'] * 20,
        'sales': np.random.randint(100, 1000, 120),
        'region': ['North', 'South', 'East', 'West'] * 30,
        'month': ['Jan', 'Feb', 'Mar'] * 40
    }
    
    df = pd.DataFrame(data)
    
    print(f"📈 Dataset shape: {df.shape}")
    print(f"📋 Dataset head:")
    print(df.head())
    
    # Basic analysis
    print(f"
🔢 Sales statistics:")
    print(df['sales'].describe())
    
    print(f"
🌍 Sales by region:")
    print(df.groupby('region')['sales'].sum())
    
    print(f"
📅 Sales by month:")
    print(df.groupby('month')['sales'].mean())
    
    return df

if __name__ == "__main__":
    df = analyze_sales_data()
    print("✅ Data analysis completed successfully!")