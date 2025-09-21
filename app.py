def prepare_data(self):
“”“Calculate current price and set tolerance, handle different column formats”””
# Standardize column names for all timeframes
for timeframe, df in self.timeframe_data.items():
df_copy = df.copy()

```
        # Handle different column name variations (case-insensitive)
        column_mapping = {}
        existing_cols = [col.lower() for col in df_copy.columns]
        
        # Map columns to standard names
        for col in df_copy.columns:
            col_lower = col.lower().strip()
            if col_lower in ['open', 'o']:
                column_mapping[col] = 'Open'
            elif col_lower in ['high', 'h']:
                column_mapping[col] = 'High'
            elif col_lower in ['low', 'l']:
                column_mapping[col] = 'Low'
            elif col_lower in ['close', 'c']:
                column_mapping[col] = 'Close'
            elif col_lower in ['volume', 'vol', 'v']:
                column_mapping[col] = 'Volume'
        
        # Apply column mapping
        if column_mapping:
            df_copy.rename(columns=column_mapping, inplace=True)
        
        # Check for required columns after mapping
        required_cols = ['Open', 'High', 'Low', 'Close']
        available_cols = df_copy.columns.tolist()
        missing_cols = [col for col in required_cols if col not in available_cols]
        
        if missing_cols:
            print(f"Available columns in {timeframe}: {available_cols}")
            print(f"Missing columns in {timeframe}: {missing_cols}")
            raise ValueError(f"Missing required columns in {timeframe}: {missing_cols}. Available: {available_cols}")
        
        # Clean data
        df_copy.dropna(inplace=True)
        self.timeframe_data[timeframe] = df_copy
    
    # Get current price from the primary (first) timeframe
    primary_timeframe = list(self.timeframe_data.keys())[0]
    self.current_price = self.timeframe_data[primary_timeframe]['Close'].iloc[-1]
    
    # Calculate tolerance based on current price
    self.tolerance = self.current_price * self.tolerance_percentage
    
    print(f"Current price: ${self.current_price:.2f}")
    print(f"Tolerance: {self.tolerance_percentage*100:.3f}% = ${self.tolerance:.3f}")
    print(f"Grouping method: {self.grouping_method}")
    print(f"Timeframes loaded: {list(self.timeframe_data.keys())}")
```
