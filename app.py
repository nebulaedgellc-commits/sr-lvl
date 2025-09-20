from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import numpy as np
from collections import defaultdict
import io
import os

app = Flask(__name__)

class MultiTimeframeSRFinder:
    def __init__(self, timeframe_data, min_touches=2, tolerance_percentage=0.01, grouping_method="conservative"):
        """
        Multi-timeframe Support & Resistance finder
        
        Parameters:
        timeframe_data: dict like {'1D': df1, '4H': df2, '1H': df3}
        min_touches: Minimum touches needed across all timeframes (flexible: 2-6)
        tolerance_percentage: Tolerance as percentage of current price (flexible: 0.005-0.05)
        grouping_method: "conservative" (less grouping) or "aggressive" (more grouping)
        """
        self.timeframe_data = timeframe_data
        self.min_touches = min_touches
        self.tolerance_percentage = tolerance_percentage / 100.0
        self.grouping_method = grouping_method
        self.timeframe_weights = {'1D': 3, '4H': 2, '1H': 1}
        self.prepare_data()
        self.timeframe_data = timeframe_data
        self.min_touches = min_touches
        self.tolerance_percentage = tolerance_percentage / 100.0  # Convert to decimal
        self.timeframe_weights = {'1D': 3, '4H': 2, '1H': 1}  # Weight importance
        self.prepare_data()
        
    def prepare_data(self):
        """Calculate current price and set tolerance, handle different column formats"""
        # Standardize column names for all timeframes
        for timeframe, df in self.timeframe_data.items():
            df_copy = df.copy()
            
            # Handle different column name variations
            column_mapping = {}
            for col in df_copy.columns:
                col_lower = col.lower()
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
            
            # Rename columns to standard format
            if column_mapping:
                df_copy.rename(columns=column_mapping, inplace=True)
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df_copy.columns]
            if missing_cols:
                # Try alternative column names
                alt_mapping = {
                    'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                    'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close'
                }
                for old_name, new_name in alt_mapping.items():
                    if old_name in df_copy.columns and new_name in missing_cols:
                        df_copy.rename(columns={old_name: new_name}, inplace=True)
                        missing_cols.remove(new_name)
                        
                if missing_cols:
                    raise ValueError(f"Missing required columns in {timeframe}: {missing_cols}")
            
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
    
    def group_prices_conservative(self, prices, level_type, timeframe, weight):
        """
        CONSERVATIVE: Less aggressive grouping - preserves more distinct levels
        Uses closest price in group for distance check
        """
        if not prices:
            return []
        
        sorted_prices = sorted(prices)
        groups = []
        current_group = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            # Check distance from CLOSEST price in current group (less aggressive)
            min_distance = min(abs(price - group_price) for group_price in current_group)
            
            if min_distance <= self.tolerance:
                current_group.append(price)
            else:
                if len(current_group) >= 1:
                    groups.append(current_group.copy())
                current_group = [price]
        
        if len(current_group) >= 1:
            groups.append(current_group)
        
        return self.convert_groups_to_levels(groups, level_type, timeframe, weight)
    
    def group_prices_aggressive(self, prices, level_type, timeframe, weight):
        """
        AGGRESSIVE: More grouping - consolidates similar levels
        Uses group center (average) for distance check
        """
        if not prices:
            return []
        
        sorted_prices = sorted(prices)
        groups = []
        current_group = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            # Check distance from group CENTER (more aggressive grouping)
            group_center = sum(current_group) / len(current_group)
            
            if abs(price - group_center) <= self.tolerance:
                current_group.append(price)
            else:
                if len(current_group) >= 1:
                    groups.append(current_group.copy())
                current_group = [price]
        
        if len(current_group) >= 1:
            groups.append(current_group)
        
        return self.convert_groups_to_levels(groups, level_type, timeframe, weight)
    
    def convert_groups_to_levels(self, groups, level_type, timeframe, weight):
        """Convert price groups to level objects"""
        levels = []
        for group in groups:
            if len(group) >= 1:
                # Use median for better stability, fallback to mean for small groups
                if len(group) >= 3:
                    level_price = np.median(group)
                else:
                    level_price = sum(group) / len(group)
                
                levels.append({
                    'level': level_price,
                    'type': level_type,
                    'touches': len(group),
                    'timeframe': timeframe,
                    'weight': weight,
                    'weighted_touches': len(group) * weight,
                    'original_prices': group,
                    'price_range': f"${min(group):.2f}-${max(group):.2f}" if len(group) > 1 else f"${group[0]:.2f}"
                })
        
        return levels
    
    def find_levels_for_timeframe(self, timeframe, df):
        """Find support and resistance levels for a single timeframe using selected algorithm"""
        weight = self.timeframe_weights.get(timeframe, 1)
        
        # Choose grouping method based on user selection
        if self.grouping_method == "conservative":
            resistance_levels = self.group_prices_conservative(df['High'].tolist(), 'Resistance', timeframe, weight)
            support_levels = self.group_prices_conservative(df['Low'].tolist(), 'Support', timeframe, weight)
        else:  # aggressive
            resistance_levels = self.group_prices_aggressive(df['High'].tolist(), 'Resistance', timeframe, weight)
            support_levels = self.group_prices_aggressive(df['Low'].tolist(), 'Support', timeframe, weight)
        
        return resistance_levels + support_levels
    
    def combine_multi_timeframe_levels(self):
        """Combine levels from all timeframes and find the strongest ones"""
        all_levels = []
        
        # Get levels from each timeframe
        for timeframe, df in self.timeframe_data.items():
            tf_levels = self.find_levels_for_timeframe(timeframe, df)
            all_levels.extend(tf_levels)
        
        # Group similar levels across timeframes using price percentage tolerance
        grouped_levels = self.group_similar_levels(all_levels)
        
        # Filter by minimum combined touches
        strong_levels = []
        for group in grouped_levels:
            total_touches = sum(level['touches'] for level in group)
            total_weighted_touches = sum(level['weighted_touches'] for level in group)
            
            if total_touches >= self.min_touches:
                # Use weighted average for final level price
                weighted_sum = sum(level['level'] * level['weighted_touches'] for level in group)
                final_level = weighted_sum / total_weighted_touches
                
                # Determine type (majority vote)
                types = [level['type'] for level in group]
                final_type = max(set(types), key=types.count)
                
                # Get timeframes involved
                timeframes = list(set(level['timeframe'] for level in group))
                
                strong_levels.append({
                    'level': final_level,
                    'type': final_type,
                    'touches': total_touches,
                    'weighted_touches': total_weighted_touches,
                    'timeframes': timeframes,
                    'timeframe_count': len(timeframes)
                })
        
        # Sort by weighted touches and timeframe count (multi-timeframe levels are stronger)
        strong_levels.sort(key=lambda x: (x['timeframe_count'], x['weighted_touches']), reverse=True)
        
    def test_tolerance_grouping(self, example_prices, tolerance_percentage):
        """
        Test function to demonstrate how price grouping works
        Example: [177.19, 177.24, 177.30, 177.90] -> groups based on tolerance
        """
        print(f"\n=== TOLERANCE GROUPING TEST ===")
        print(f"Example prices: {example_prices}")
        print(f"Tolerance: {tolerance_percentage}% of current price")
        
        # Set tolerance for testing
        if example_prices:
            test_price = max(example_prices)  # Use highest price as reference
            tolerance = test_price * (tolerance_percentage / 100.0)
            print(f"Reference price: ${test_price:.2f}")
            print(f"Calculated tolerance: ${tolerance:.3f}")
        else:
            return []
        
        # Sort prices
        sorted_prices = sorted(example_prices)
        groups = []
        current_group = [sorted_prices[0]]
        
        print(f"\nGrouping process:")
        print(f"Starting with first price: ${sorted_prices[0]:.2f}")
        
        for i, price in enumerate(sorted_prices[1:], 1):
            # Calculate group center
            group_center = sum(current_group) / len(current_group)
            distance = abs(price - group_center)
            
            print(f"\nPrice ${price:.2f}:")
            print(f"  Current group center: ${group_center:.3f}")
            print(f"  Distance: ${distance:.3f}")
            print(f"  Tolerance: ${tolerance:.3f}")
            
            if distance <= tolerance:
                current_group.append(price)
                print(f"  ✓ Added to current group: {[f'${p:.2f}' for p in current_group]}")
            else:
                print(f"  ✗ Too far, starting new group")
                groups.append(current_group.copy())
                current_group = [price]
                print(f"  New group: [${price:.2f}]")
        
        # Add final group
        groups.append(current_group)
        
        print(f"\n=== FINAL GROUPS ===")
        for i, group in enumerate(groups, 1):
            avg_price = sum(group) / len(group)
            price_range = f"${min(group):.2f} - ${max(group):.2f}"
            print(f"Group {i}: {len(group)} prices ({price_range}) → Level: ${avg_price:.2f}")
        
        return groups
    
    def group_similar_levels(self, all_levels):
        """Group levels that are within tolerance of each other"""
        if not all_levels:
            return []
        
        # Sort levels by price
        sorted_levels = sorted(all_levels, key=lambda x: x['level'])
        groups = []
        current_group = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Check if this level is close to any level in current group
            if any(abs(level['level'] - group_level['level']) <= self.tolerance 
                   for group_level in current_group):
                current_group.append(level)
            else:
                # Start new group
                groups.append(current_group)
                current_group = [level]
        
        groups.append(current_group)  # Add the last group
        
        return groups
    
    def get_levels_only(self):
        """Get only the price levels as comma-separated string"""
        levels = self.combine_multi_timeframe_levels()
        level_prices = [f"{level['level']:.2f}" for level in levels]
        return ",".join(level_prices)
    
    def get_detailed_results(self):
        """Get detailed results for analysis"""
        levels = self.combine_multi_timeframe_levels()
        return {
            'levels_csv': self.get_levels_only(),
            'total_count': len(levels),
            'detailed_levels': levels,
            'timeframes_used': list(self.timeframe_data.keys()),
            'tolerance_info': {
                'percentage': self.tolerance_percentage * 100,
                'dollar_amount': self.tolerance,
                'current_price': self.current_price
            }
        }

# HTML template for multi-timeframe interface with percentage tolerance
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Timeframe S&R Level Finder</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 900px; 
            margin: 30px auto; 
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container { 
            background: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #333; 
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-style: italic;
        }
        .form-group { 
            margin-bottom: 20px; 
        }
        .file-group {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        .file-group label {
            width: 80px;
            font-weight: bold;
            margin-right: 15px;
        }
        .file-group input {
            flex: 1;
        }
        label { 
            display: block; 
            margin-bottom: 5px; 
            font-weight: bold;
        }
        input, select { 
            width: 100%; 
            padding: 10px; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            box-sizing: border-box;
        }
        .file-group input {
            width: auto;
        }
        button { 
            background: #007bff; 
            color: white; 
            padding: 12px 30px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            font-size: 16px;
            width: 100%;
            margin-top: 20px;
        }
        button:hover { 
            background: #0056b3; 
        }
        .result { 
            margin-top: 30px; 
            padding: 20px; 
            background: #f8f9fa; 
            border-radius: 5px; 
            border-left: 4px solid #007bff;
        }
        .error { 
            background: #f8d7da; 
            color: #721c24; 
            border-left-color: #dc3545;
        }
        .levels-output {
            font-family: monospace;
            font-size: 14px;
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            word-break: break-all;
            margin-top: 10px;
        }
        .info {
            background: #d1ecf1;
            color: #0c5460;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .file-section {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
            background: #fafafa;
        }
        .optional {
            color: #666;
            font-size: 12px;
        }
        .tolerance-section {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .tolerance-examples {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .details {
            margin-top: 15px;
            font-size: 14px;
        }
        .details table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .details th, .details td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .details th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Timeframe S&R Level Finder</h1>
        <div class="subtitle">Upload multiple timeframes for stronger support & resistance levels</div>
        
        <div class="info">
            <strong>How it works:</strong><br>
            • Upload 1D OHLC data (required) + optional 4H and 1H data<br>
            • Higher timeframes get more weight (1D=3x, 4H=2x, 1H=1x)<br>
            • Levels confirmed across multiple timeframes are prioritized<br>
            • <strong>NEW:</strong> Tolerance based on percentage of current stock price
        </div>

        <form method="post" enctype="multipart/form-data">
            <div class="file-section">
                <h3>Upload Timeframe Data</h3>
                <div class="file-group">
                    <label for="file_1d">1D:</label>
                    <input type="file" name="file_1d" accept=".csv" required>
                    <span style="margin-left: 10px; color: #28a745; font-weight: bold;">Required</span>
                </div>
                
                <div class="file-group">
                    <label for="file_4h">4H:</label>
                    <input type="file" name="file_4h" accept=".csv">
                    <span class="optional" style="margin-left: 10px;">Optional</span>
                </div>
                
                <div class="file-group">
                    <label for="file_1h">1H:</label>
                    <input type="file" name="file_1h" accept=".csv">
                    <span class="optional" style="margin-left: 10px;">Optional</span>
                </div>
            </div>

            <div class="tolerance-section">
                <h3>🎯 Analysis Settings</h3>
                
                <div class="form-group">
                    <label for="grouping_method">Grouping Method:</label>
                    <select name="grouping_method">
                        <option value="conservative" selected>Conservative - Preserves more distinct levels (recommended)</option>
                        <option value="aggressive">Aggressive - Groups similar levels more (for noisy data)</option>
                    </select>
                    <div class="tolerance-examples">
                        <strong>Conservative:</strong> Keeps 179.88 and 179.38 as separate levels<br>
                        <strong>Aggressive:</strong> Groups 179.88 and 179.38 into single level ~179.63
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="tolerance_percentage">Tolerance (% of current stock price):</label>
                    <select name="tolerance_percentage">
                        <option value="0.005">0.005% - Ultra Tight (finds micro-levels)</option>
                        <option value="0.01" selected>0.01% - Tight (recommended for precision)</option>
                        <option value="0.015">0.015% - Moderate</option>
                        <option value="0.02">0.02% - Balanced</option>
                        <option value="0.03">0.03% - Loose</option>
                        <option value="0.05">0.05% - Very Loose (consolidates levels)</option>
                    </select>
                    <div class="tolerance-examples">
                        <strong>Examples:</strong> For $180 stock → 0.01% = $0.018 tolerance | For $50 stock → 0.01% = $0.005 tolerance
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="min_touches">Minimum Total Touches (across all timeframes):</label>
                    <select name="min_touches">
                        <option value="2" selected>2 - Very Sensitive (captures more levels)</option>
                        <option value="3">3 - Sensitive</option>
                        <option value="4">4 - Balanced</option>
                        <option value="5">5 - Conservative</option>
                        <option value="6">6 - Very Conservative (only strongest levels)</option>
                    </select>
                </div>
            </div>
            
            <button type="submit">Find Multi-Timeframe S&R Levels</button>
        </form>

        {% if result %}
        <div class="result">
            <h3>🎯 Results:</h3>
            <p><strong>Timeframes used:</strong> {{ result.timeframes_used | join(', ') }}</p>
            <p><strong>Total strong levels found:</strong> {{ result.total_count }}</p>
            <p><strong>Grouping method:</strong> {{ result.grouping_method.title() }}</p>
            <p><strong>Tolerance:</strong> {{ "%.3f"|format(result.tolerance_info.percentage) }}% of price = ${{ "%.3f"|format(result.tolerance_info.dollar_amount) }}</p>
            <p><strong>Current price:</strong> ${{ "%.2f"|format(result.tolerance_info.current_price) }}</p>
            <p><strong>Settings:</strong> Min touches: {{ result.min_touches }}</p>
            
            <strong>📋 Levels for TradingView (copy this):</strong>
            <div class="levels-output">{{ result.levels_csv }}</div>
            
            <div class="details">
                <strong>📊 Level Details:</strong>
                <table>
                    <tr>
                        <th>Level</th>
                        <th>Type</th>
                        <th>Total Touches</th>
                        <th>Timeframes</th>
                        <th>Strength Score</th>
                    </tr>
                    {% for level in result.detailed_levels[:10] %}
                    <tr>
                        <td>${{ "%.2f"|format(level.level) }}</td>
                        <td>{{ level.type }}</td>
                        <td>{{ level.touches }}</td>
                        <td>{{ level.timeframes | join(', ') }}</td>
                        <td>{{ level.weighted_touches }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% if result.detailed_levels|length > 10 %}
                <p><em>Showing top 10 levels...</em></p>
                {% endif %}
            </div>
        </div>
        {% endif %}

        {% if error %}
        <div class="result error">
            <h3>❌ Error:</h3>
            <p>{{ error }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE)
    
    try:
        # Get form data
        min_touches = int(request.form.get('min_touches', 2))
        tolerance_percentage = float(request.form.get('tolerance_percentage', 0.01))
        grouping_method = request.form.get('grouping_method', 'conservative')
        
        # Process uploaded files
        timeframe_data = {}
        
        # Check for required 1D file
        if 'file_1d' not in request.files or request.files['file_1d'].filename == '':
            raise ValueError("1D timeframe file is required")
        
        # Process 1D file (required)
        file_1d = request.files['file_1d']
        csv_data = file_1d.read().decode('utf-8')
        df_1d = pd.read_csv(io.StringIO(csv_data))
        
        # Validate columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df_1d.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in 1D file: {', '.join(missing_cols)}")
        
        if 'time' in df_1d.columns:
            df_1d['time'] = pd.to_datetime(df_1d['time'])
            df_1d.set_index('time', inplace=True)
        elif 'Date' in df_1d.columns:
            df_1d['Date'] = pd.to_datetime(df_1d['Date'])
            df_1d.set_index('Date', inplace=True)
        
        timeframe_data['1D'] = df_1d
        
        # Process optional 4H file
        if 'file_4h' in request.files and request.files['file_4h'].filename != '':
            try:
                file_4h = request.files['file_4h']
                csv_data_4h = file_4h.read().decode('utf-8')
                df_4h = pd.read_csv(io.StringIO(csv_data_4h))
                
                if 'time' in df_4h.columns:
                    df_4h['time'] = pd.to_datetime(df_4h['time'])
                    df_4h.set_index('time', inplace=True)
                elif 'Date' in df_4h.columns:
                    df_4h['Date'] = pd.to_datetime(df_4h['Date'])
                    df_4h.set_index('Date', inplace=True)
                
                timeframe_data['4H'] = df_4h
            except Exception as e:
                print(f"Warning: Could not process 4H file: {e}")
        
        # Process optional 1H file
        if 'file_1h' in request.files and request.files['file_1h'].filename != '':
            try:
                file_1h = request.files['file_1h']
                csv_data_1h = file_1h.read().decode('utf-8')
                df_1h = pd.read_csv(io.StringIO(csv_data_1h))
                
                if 'time' in df_1h.columns:
                    df_1h['time'] = pd.to_datetime(df_1h['time'])
                    df_1h.set_index('time', inplace=True)
                elif 'Date' in df_1h.columns:
                    df_1h['Date'] = pd.to_datetime(df_1h['Date'])
                    df_1h.set_index('Date', inplace=True)
                
                timeframe_data['1H'] = df_1h
            except Exception as e:
                print(f"Warning: Could not process 1H file: {e}")
        
        # Analyze multi-timeframe levels with percentage tolerance and grouping method
        finder = MultiTimeframeSRFinder(timeframe_data, min_touches, tolerance_percentage, grouping_method)
        results = finder.get_detailed_results()
        
        # Add form parameters to results
        results['min_touches'] = min_touches
        results['tolerance_percentage'] = tolerance_percentage
        results['grouping_method'] = grouping_method
        
        return render_template_string(HTML_TEMPLATE, result=results)
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=str(e))

@app.route('/api/analyze-multi', methods=['POST'])
def api_analyze_multi():
    """API endpoint for multi-timeframe analysis with percentage tolerance"""
    try:
        min_touches = int(request.form.get('min_touches', 2))
        tolerance_percentage = float(request.form.get('tolerance_percentage', 0.01))
        grouping_method = request.form.get('grouping_method', 'conservative')
        
        timeframe_data = {}
        
        # Process files
        if 'file_1d' not in request.files:
            return jsonify({'error': '1D file is required'}), 400
        
        # Process 1D (required)
        file_1d = request.files['file_1d']
        csv_data = file_1d.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        timeframe_data['1D'] = df
        
        # Process optional files
        for tf_key, file_key in [('4H', 'file_4h'), ('1H', 'file_1h')]:
            if file_key in request.files and request.files[file_key].filename != '':
                try:
                    file_data = request.files[file_key].read().decode('utf-8')
                    df_tf = pd.read_csv(io.StringIO(file_data))
                    if 'time' in df_tf.columns:
                        df_tf['time'] = pd.to_datetime(df_tf['time'])
                        df_tf.set_index('time', inplace=True)
                    elif 'Date' in df_tf.columns:
                        df_tf['Date'] = pd.to_datetime(df_tf['Date'])
                        df_tf.set_index('Date', inplace=True)
                    timeframe_data[tf_key] = df_tf
                except Exception as e:
                    print(f"Warning: Could not process {tf_key} file: {e}")
        
        finder = MultiTimeframeSRFinder(timeframe_data, min_touches, tolerance_percentage, grouping_method)
        results = finder.get_detailed_results()
        
        return jsonify({
            'success': True,
            'levels': results['levels_csv'],
            'count': results['total_count'],
            'timeframes_used': results['timeframes_used'],
            'tolerance_info': results['tolerance_info']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/test-grouping', methods=['GET', 'POST'])
def test_grouping():
    """Test route to demonstrate price grouping logic"""
    
    if request.method == 'GET':
        test_html = '''
        <!DOCTYPE html>
        <html>
        <head><title>Test Tolerance Grouping</title></head>
        <body style="font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px;">
            <h1>Test Tolerance Grouping Logic</h1>
            <p>Test how prices get grouped based on percentage tolerance.</p>
            
            <form method="post">
                <div style="margin-bottom: 15px;">
                    <label>Enter prices (comma-separated):</label><br>
                    <input type="text" name="prices" value="177.19, 177.24, 177.30, 177.90" style="width: 100%; padding: 10px;" />
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label>Tolerance (% of price):</label><br>
                    <select name="tolerance" style="padding: 10px;">
                        <option value="0.01">0.01%</option>
                        <option value="0.02" selected>0.02%</option>
                        <option value="0.03">0.03%</option>
                        <option value="0.05">0.05%</option>
                        <option value="0.1">0.1%</option>
                    </select>
                </div>
                
                <button type="submit" style="padding: 10px 20px; background: #007bff; color: white; border: none;">Test Grouping</button>
            </form>
        </body>
        </html>
        '''
        return test_html
    
    try:
        # Get form data
        prices_str = request.form.get('prices', '177.19, 177.24, 177.30, 177.90')
        tolerance_pct = float(request.form.get('tolerance', 0.02))
        
        # Parse prices
        prices = [float(p.strip()) for p in prices_str.split(',')]
        
        # Create a dummy finder to test grouping
        dummy_data = {'1D': pd.DataFrame({'Close': [max(prices)]})}  # Just for getting current price
        finder = MultiTimeframeSRFinder(dummy_data, min_touches=1, tolerance_percentage=tolerance_pct)
        
        # Test the grouping
        groups = finder.test_tolerance_grouping(prices, tolerance_pct)
        
        # Create result HTML
        result_html = f'''
        <!DOCTYPE html>
        <html>
        <head><title>Grouping Results</title></head>
        <body style="font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px;">
            <h1>Tolerance Grouping Results</h1>
            <p><strong>Input prices:</strong> {prices_str}</p>
            <p><strong>Tolerance:</strong> {tolerance_pct}%</p>
            <p><strong>Reference price:</strong> ${max(prices):.2f}</p>
            <p><strong>Dollar tolerance:</strong> ${max(prices) * (tolerance_pct/100):.3f}</p>
            
            <h2>Groups Found:</h2>
        '''
        
        for i, group in enumerate(groups, 1):
            avg_price = sum(group) / len(group)
            price_list = ', '.join([f'${p:.2f}' for p in group])
            price_range = f"${min(group):.2f} - ${max(group):.2f}" if len(group) > 1 else f"${group[0]:.2f}"
            
            result_html += f'''
            <div style="background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px;">
                <strong>Group {i}:</strong> {len(group)} price{"s" if len(group) > 1 else ""}<br>
                <strong>Prices:</strong> {price_list}<br>
                <strong>Range:</strong> {price_range}<br>
                <strong>Final Level:</strong> <span style="background: #007bff; color: white; padding: 2px 8px; border-radius: 3px;">${avg_price:.2f}</span>
            </div>
            '''
        
        result_html += '''
            <br><a href="/test-grouping" style="background: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Test Again</a>
            <a href="/" style="background: #6c757d; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-left: 10px;">Back to Main App</a>
        </body>
        </html>
        '''
        
        return result_html
        
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
