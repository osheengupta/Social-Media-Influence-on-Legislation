import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime, timedelta

def analyze_tweet_metrics_legislative_impact_extended(topic_analysis):
    """
    Analyze the relationship between tweet metrics (aggressiveness and sentiment)
    and legislative activity with extended time lags (30-days, 60-days, 90-days).
    
    Parameters:
    -----------
    topic_analysis : dictionary
        Dictionary containing the analyzed data for each topic
    
    Returns:
    --------
    dict : Dictionary containing regression results
    """
    print("Analyzing the impact of tweet metrics on future legislative activity with extended time lags...")
    
    # Extract the data for each topic
    gas_emissions_data = topic_analysis['Seriousness of Gas Emissions']
    global_warming_data = topic_analysis['Ideological Positions on Global Warming']
    
    # Get tweet data for each topic
    tweets_gas = gas_emissions_data['tweets']
    tweets_warming = global_warming_data['tweets']
    
    # Get bill data for each topic
    bills_gas = gas_emissions_data['bills']
    bills_warming = global_warming_data['bills']
    
    # Rename date columns for consistency
    tweets_gas = tweets_gas.rename(columns={'created_at': 'date'})
    tweets_warming = tweets_warming.rename(columns={'created_at': 'date'})
    
    if not bills_gas.empty:
        bills_gas = bills_gas.rename(columns={'Date of Introduction': 'date'})
    
    if not bills_warming.empty:
        bills_warming = bills_warming.rename(columns={'Date of Introduction': 'date'})
    
    # Merge the tweet datasets on date - drop NA values instead of filling with zeros
    combined_tweets = pd.merge(
        tweets_gas, 
        tweets_warming, 
        on='date', 
        how='outer',
        suffixes=('_gas', '_warming')
    ).dropna()
    
    # Calculate combined metrics
    combined_tweets['total_volume'] = combined_tweets['id_gas'] + combined_tweets['id_warming']
    
    # Calculate weighted averages for aggressiveness and sentiment
    combined_tweets['combined_aggressiveness'] = (
        (combined_tweets['aggressiveness_gas'] * combined_tweets['id_gas']) + 
        (combined_tweets['aggressiveness_warming'] * combined_tweets['id_warming'])
    ) / combined_tweets['total_volume']
    
    combined_tweets['combined_sentiment'] = (
        (combined_tweets['sentiment_gas'] * combined_tweets['id_gas']) + 
        (combined_tweets['sentiment_warming'] * combined_tweets['id_warming'])
    ) / combined_tweets['total_volume']
    
    # Sort by date
    combined_tweets = combined_tweets.sort_values('date')
    
    # Truncate data to 2009-2020 period
    start_date = pd.to_datetime('2009-01-01')
    end_date = pd.to_datetime('2020-01-01')
    combined_tweets = combined_tweets[(combined_tweets['date'] >= start_date) & (combined_tweets['date'] < end_date)]
    
    # Merge the bill datasets on date - also drop NA values
    if not bills_gas.empty and not bills_warming.empty:
        combined_bills = pd.merge(
            bills_gas, 
            bills_warming, 
            on='date', 
            how='outer',
            suffixes=('_gas', '_warming')
        ).dropna()
        
        # Calculate total bills
        combined_bills['total_bills'] = combined_bills['Legislation Number_gas'] + combined_bills['Legislation Number_warming']
        
        # Sort by date
        combined_bills = combined_bills.sort_values('date')
        
        # Truncate bill data to 2009-2020 period
        combined_bills = combined_bills[(combined_bills['date'] >= start_date) & (combined_bills['date'] < end_date)]
    else:
        combined_bills = pd.DataFrame(columns=['date', 'total_bills'])
    
    # Resample data to daily frequency and drop missing values instead of filling
    daily_tweets = combined_tweets.set_index('date').resample('D').mean()
    
    # Create a daily bills dataset
    if not combined_bills.empty:
        daily_bills = combined_bills.set_index('date').resample('D').sum()
    else:
        daily_bills = pd.DataFrame(index=daily_tweets.index, columns=['total_bills'])
    
    # Merge daily tweets and bills - drop NA values
    daily_data = pd.merge(
        daily_tweets[['combined_aggressiveness', 'combined_sentiment', 'total_volume']], 
        daily_bills[['total_bills']], 
        left_index=True, 
        right_index=True, 
        how='left'
    ).dropna()
    
    # Create time-shifted features with extended lags
    # Standard lags
    daily_data['next_day_bills'] = daily_data['total_bills'].shift(-1)
    daily_data['next_week_bills'] = daily_data['total_bills'].shift(-7)
    
    # Extended lags - 30 days, 60 days, 90 days
    daily_data['next_30days_bills'] = daily_data['total_bills'].shift(-30)
    daily_data['next_60days_bills'] = daily_data['total_bills'].shift(-60)
    daily_data['next_90days_bills'] = daily_data['total_bills'].shift(-90)
    
    # Drop rows with NaN values (due to shift)
    daily_data = daily_data.dropna()
    
    # Use all data points, including those with zero legislative activity
    next_day_data = daily_data.copy()
    next_week_data = daily_data.copy()
    next_30days_data = daily_data.copy()
    next_60days_data = daily_data.copy()
    next_90days_data = daily_data.copy()
    
    # Print some diagnostic information
    print(f"Total days in dataset: {len(daily_data)}")
    print(f"Days with non-zero next-day legislative activity: {len(daily_data[daily_data['next_day_bills'] > 0])}")
    print(f"Days with non-zero next-week legislative activity: {len(daily_data[daily_data['next_week_bills'] > 0])}")
    print(f"Days with non-zero next-30-days legislative activity: {len(daily_data[daily_data['next_30days_bills'] > 0])}")
    print(f"Days with non-zero next-60-days legislative activity: {len(daily_data[daily_data['next_60days_bills'] > 0])}")
    print(f"Days with non-zero next-90-days legislative activity: {len(daily_data[daily_data['next_90days_bills'] > 0])}")
    
    # Store results
    results = {}
    
    # Increase font sizes by 50% for better readability
    plt.rcParams.update({
        'font.size': 12 * 1.5,
        'axes.labelsize': 12 * 1.5,
        'axes.titlesize': 14 * 1.5,
        'xtick.labelsize': 10 * 1.5,
        'ytick.labelsize': 10 * 1.5,
        'legend.fontsize': 10 * 1.5
    })
    
    # Function to perform regression and visualization for a specific time lag
    def analyze_time_lag(data, target_col, time_description, results_dict, color='#d62728'):
        print(f"\nRegression: Today's Metrics vs. {time_description} Legislative Activity")
        
        # 1. Aggressiveness regression
        X_aggr = sm.add_constant(data['combined_aggressiveness'])
        y = data[target_col]
        
        # Fit the model
        model_aggr = sm.OLS(y, X_aggr).fit()
        
        # Print summary
        print("\nAggressiveness vs. Legislative Activity:")
        print(model_aggr.summary().tables[1])
        
        # Store results
        results_dict[f'aggressiveness_{target_col}'] = {
            'r_squared': model_aggr.rsquared,
            'p_values': model_aggr.pvalues.to_dict(),
            'coefficients': model_aggr.params.to_dict()
        }
        
        # Visualize the relationship
        plt.figure(figsize=(12, 8))
        plt.scatter(data['combined_aggressiveness'], data[target_col], 
                    alpha=0.5, color=color)
        
        # Add regression line
        x_range = np.linspace(data['combined_aggressiveness'].min(), 
                              data['combined_aggressiveness'].max(), 100)
        X_plot = sm.add_constant(x_range)
        plt.plot(x_range, model_aggr.predict(X_plot), color, linewidth=2)
        
        plt.xlabel('Tweet Aggressiveness Today')
        plt.ylabel(f'Legislative Activity {time_description}')
        plt.title(f'Relationship Between Today\'s Tweet Aggressiveness and {time_description} Legislative Activity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'aggressiveness_{target_col}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Sentiment regression
        X_sent = sm.add_constant(data['combined_sentiment'])
        
        # Fit the model
        model_sent = sm.OLS(y, X_sent).fit()
        
        # Print summary
        print("\nSentiment vs. Legislative Activity:")
        print(model_sent.summary().tables[1])
        
        # Store results
        results_dict[f'sentiment_{target_col}'] = {
            'r_squared': model_sent.rsquared,
            'p_values': model_sent.pvalues.to_dict(),
            'coefficients': model_sent.params.to_dict()
        }
        
        # Visualize the relationship
        plt.figure(figsize=(12, 8))
        plt.scatter(data['combined_sentiment'], data[target_col], 
                    alpha=0.5, color='#2ca02c')
        
        # Add regression line
        x_range = np.linspace(data['combined_sentiment'].min(), 
                              data['combined_sentiment'].max(), 100)
        X_plot = sm.add_constant(x_range)
        plt.plot(x_range, model_sent.predict(X_plot), 'g', linewidth=2)
        
        plt.xlabel('Tweet Sentiment Today')
        plt.ylabel(f'Legislative Activity {time_description}')
        plt.title(f'Relationship Between Today\'s Tweet Sentiment and {time_description} Legislative Activity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'sentiment_{target_col}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Multiple regression (both metrics)
        X_combined = sm.add_constant(data[['combined_aggressiveness', 'combined_sentiment']])
        
        # Fit the model
        model_combined = sm.OLS(y, X_combined).fit()
        
        # Print summary
        print("\nCombined Metrics vs. Legislative Activity:")
        print(model_combined.summary().tables[1])
        
        # Store results
        results_dict[f'combined_metrics_{target_col}'] = {
            'r_squared': model_combined.rsquared,
            'p_values': model_combined.pvalues.to_dict(),
            'coefficients': model_combined.params.to_dict()
        }
        
        return model_aggr, model_sent, model_combined
    
    # Analyze each time lag
    print("\n--- STANDARD TIME LAGS ---")
    analyze_time_lag(next_day_data, 'next_day_bills', 'Next Day', results, '#d62728')
    analyze_time_lag(next_week_data, 'next_week_bills', 'Next Week', results, '#9467bd')
    
    print("\n--- EXTENDED TIME LAGS ---")
    analyze_time_lag(next_30days_data, 'next_30days_bills', 'Next 30 Days', results, '#1f77b4')
    analyze_time_lag(next_60days_data, 'next_60days_bills', 'Next 60 Days', results, '#ff7f0e')
    analyze_time_lag(next_90days_data, 'next_90days_bills', 'Next 90 Days', results, '#2ca02c')
    
    # Create a summary visualization comparing coefficients across time lags
    time_lags = ['Next Day', 'Next Week', 'Next 30 Days', 'Next 60 Days', 'Next 90 Days']
    aggr_coefs = [
        results['aggressiveness_next_day_bills']['coefficients']['combined_aggressiveness'],
        results['aggressiveness_next_week_bills']['coefficients']['combined_aggressiveness'],
        results['aggressiveness_next_30days_bills']['coefficients']['combined_aggressiveness'],
        results['aggressiveness_next_60days_bills']['coefficients']['combined_aggressiveness'],
        results['aggressiveness_next_90days_bills']['coefficients']['combined_aggressiveness']
    ]
    
    sent_coefs = [
        results['sentiment_next_day_bills']['coefficients']['combined_sentiment'],
        results['sentiment_next_week_bills']['coefficients']['combined_sentiment'],
        results['sentiment_next_30days_bills']['coefficients']['combined_sentiment'],
        results['sentiment_next_60days_bills']['coefficients']['combined_sentiment'],
        results['sentiment_next_90days_bills']['coefficients']['combined_sentiment']
    ]
    
    # Create a bar chart comparing coefficients
    plt.figure(figsize=(14, 10))
    
    x = np.arange(len(time_lags))
    width = 0.35
    
    plt.bar(x - width/2, aggr_coefs, width, label='Aggressiveness Coefficient', color='#d62728')
    plt.bar(x + width/2, sent_coefs, width, label='Sentiment Coefficient', color='#2ca02c')
    
    plt.xlabel('Time Lag')
    plt.ylabel('Coefficient Value')
    #plt.title('Impact of Tweet Metrics on Legislative Activity Across Different Time Lags')
    plt.xticks(x, time_lags)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add value labels on the bars
    for i, v in enumerate(aggr_coefs):
        plt.text(i - width/2, v + (1 if v > 0 else -3), f'{v:.2f}', 
                 ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')
    
    for i, v in enumerate(sent_coefs):
        plt.text(i + width/2, v + (1 if v > 0 else -3), f'{v:.2f}', 
                 ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('coefficient_comparison_across_time_lags.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a line chart showing how the coefficients change over time
    plt.figure(figsize=(14, 8))
    
    # Convert time lags to numeric values for x-axis
    x_values = [1, 7, 30, 60, 90]  # days
    
    plt.plot(x_values, aggr_coefs, 'o-', linewidth=2, label='Aggressiveness Coefficient', color='#d62728')
    plt.plot(x_values, sent_coefs, 's-', linewidth=2, label='Sentiment Coefficient', color='#2ca02c')
    
    plt.xlabel('Time Lag (Days)')
    plt.ylabel('Coefficient Value')
    plt.title('Evolution of Tweet Metrics Impact Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Set x-axis to log scale to better visualize the different time scales
    plt.xscale('log')
    plt.xticks(x_values, [str(x) for x in x_values])
    
    # Add value labels on the points
    for i, v in enumerate(aggr_coefs):
        plt.text(x_values[i], v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    for i, v in enumerate(sent_coefs):
        plt.text(x_values[i], v - 0.5, f'{v:.2f}', ha='center', va='top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('coefficient_evolution_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary of findings
    print("\n--- SUMMARY OF FINDINGS ---")
    print("Coefficient values across different time lags:")
    print("\nAggressiveness coefficients:")
    for i, lag in enumerate(time_lags):
        print(f"{lag}: {aggr_coefs[i]:.4f} (p-value: {results[f'aggressiveness_next_day_bills' if i==0 else f'aggressiveness_next_week_bills' if i==1 else f'aggressiveness_next_30days_bills' if i==2 else f'aggressiveness_next_60days_bills' if i==3 else 'aggressiveness_next_90days_bills']['p_values']['combined_aggressiveness']:.4f})")
    
    print("\nSentiment coefficients:")
    for i, lag in enumerate(time_lags):
        print(f"{lag}: {sent_coefs[i]:.4f} (p-value: {results[f'sentiment_next_day_bills' if i==0 else f'sentiment_next_week_bills' if i==1 else f'sentiment_next_30days_bills' if i==2 else f'sentiment_next_60days_bills' if i==3 else 'sentiment_next_90days_bills']['p_values']['combined_sentiment']:.4f})")
    
    return results

# Example usage:
# regression_results = analyze_tweet_metrics_legislative_impact_extended(topic_analysis)
