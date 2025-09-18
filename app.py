from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import numpy as np
from collections import defaultdict
import io
import os

app = Flask(**name**)

class MultiTimeframeSRFinder:
def **init**(self, timeframe_data, min_touches=4, atr_tolerance_pct=0.1):

# Multi-timeframe Support & Resistance finder
# Parameters:
# timeframe_data: dict like {'1D': df1, '4H': df2, '1H': df3}
# min_touches: Minimum touches needed across all timeframes
# atr_tolerance_pct: Tolerance as % of ATR for grouping levels

    self.timeframe_data = timeframe_data
    self.min_touches = min_touches
    self.atr_tolerance_pct = atr_tolerance_pct / 100.0
    self.timeframe_weights = {'1D': 3, '4H': 2, '1H': 1}  # Weight importance
    self.prepare_data()
    
def prepare_data(self):
    # Calculate ATR for each timeframe and determine overall tolerance
    self.atr_values = {}
    
    for timeframe, df in self.timeframe_data.items():
        df_copy = df.copy()
        
        # Calculate ATR
        high_low = df_copy['High'] - df_copy['Low']
        high_close = np.abs(df_copy['High'] - df_copy['Close'].shift())
        low_close = np.abs(df_copy['Low'] - df_copy['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df_copy['ATR'] = true_range.rolling(window=14).mean()
        df_copy.dropna(inplace=True)
        
        self.timeframe_data[timeframe] = df_copy
        self.atr_values[timeframe] = df_copy['ATR'].mean()
    
    # Use the highest timeframe ATR for tolerance (usually 1D)
    primary_atr = max(self.atr_values.values())
    self.tolerance = primary_atr * self.atr_tolerance_pct
    
    print(f"ATR Values: {self.atr_values}")
    print(f"Using tolerance: {self.tolerance:.4f}")

def find_levels_for_timeframe(self, timeframe, df):
    """Find support and resistance levels for a single timeframe"""
    levels = []
    weight = self.timeframe_weights.get(timeframe, 1)
    
    # Find resistance levels (highs)
    resistance_counts = defaultdict(list)
    for i, high_price in enumerate(df['High']):
        price_level = round(high_price, 2)
        resistance_counts[price_level].append((i, high_price))
    
    for level, touches in resistance_counts.items():
        if len(touches) >= 2:  # Lower threshold per timeframe
            levels.append({
                'level': level,
                'type': 'Resistance',
                'touches': len(touches),
                'timeframe': timeframe,
                'weight': weight,
                'weighted_touches': len(touches) * weight
            })
    
    # Find support levels (lows)  
    support_counts = defaultdict(list)
    for i, low_price in enumerate(df['Low']):
        price_level = round(low_price, 2)
        support_counts[price_level].append((i, low_price))
    
    for level, touches in support_counts.items():
        if len(touches) >= 2:  # Lower threshold per timeframe
            levels.append({
                'level': level,
                'type': 'Support', 
                'touches': len(touches),
                'timeframe': timeframe,
                'weight': weight,
                'weighted_touches': len(touches) * weight
            })
    
    return levels

def combine_multi_timeframe_levels(self):
    """Combine levels from all timeframes and find the strongest ones"""
    all_levels = []
    
    # Get levels from each timeframe
    for timeframe, df in self.timeframe_data.items():
        tf_levels = self.find_levels_for_timeframe(timeframe, df)
        all_levels.extend(tf_levels)
    
    # Group similar levels across timeframes
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
    
    return strong_levels

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
        'atr_values': self.atr_values
    }

# HTML template for multi-timeframe interface

HTML_TEMPLATE = 
""" <!DOCTYPE html>

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
        • Get the strongest levels as comma-separated values for TradingView
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
        
        <div class="form-group">
            <label for="min_touches">Minimum Total Touches (across all timeframes):</label>
            <select name="min_touches">
                <option value="3">3</option>
                <option value="4" selected>4</option>
                <option value="5">5</option>
                <option value="6">6</option>
                <option value="8">8</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="atr_tolerance">ATR Tolerance (%):</label>
            <select name="atr_tolerance">
                <option value="0.05">0.05%</option>
                <option value="0.1" selected>0.1%</option>
                <option value="0.2">0.2%</option>
                <option value="0.5">0.5%</option>
            </select>
        </div>
        
        <button type="submit">Find Multi-Timeframe S&R Levels</button>
    </form>

    {% if result %}
    <div class="result">
        <h3>🎯 Results:</h3>
        <p><strong>Timeframes used:</strong> {{ result.timeframes_used | join(', ') }}</p>
        <p><strong>Total strong levels found:</strong> {{ result.total_count }}</p>
        <p><strong>Settings:</strong> Min touches: {{ result.min_touches }}, ATR tolerance: {{ result.atr_tolerance }}%</p>
        
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
                    <td>{{ "%.2f"|format(level.level) }}</td>
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
"""

# Use standard single quotes
@app.route('/', methods=['GET', 'POST'])
def index():
if request.method == ‘GET’:
return render_template_string(HTML_TEMPLATE)

try:
    # Get form data
    min_touches = int(request.form.get('min_touches', 4))
    atr_tolerance = float(request.form.get('atr_tolerance', 0.1))
    
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
    
    if 'Date' in df_1d.columns:
        df_1d['Date'] = pd.to_datetime(df_1d['Date'])
        df_1d.set_index('Date', inplace=True)
    
    timeframe_data['1D'] = df_1d
    
    # Process optional 4H file
    if 'file_4h' in request.files and request.files['file_4h'].filename != '':
        try:
            file_4h = request.files['file_4h']
            csv_data_4h = file_4h.read().decode('utf-8')
            df_4h = pd.read_csv(io.StringIO(csv_data_4h))
            
            if 'Date' in df_4h.columns:
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
            
            if 'Date' in df_1h.columns:
                df_1h['Date'] = pd.to_datetime(df_1h['Date'])
                df_1h.set_index('Date', inplace=True)
            
            timeframe_data['1H'] = df_1h
        except Exception as e:
            print(f"Warning: Could not process 1H file: {e}")
    
    # Analyze multi-timeframe levels
    finder = MultiTimeframeSRFinder(timeframe_data, min_touches, atr_tolerance)
    results = finder.get_detailed_results()
    
    # Add form parameters to results
    results['min_touches'] = min_touches
    results['atr_tolerance'] = atr_tolerance
    
    return render_template_string(HTML_TEMPLATE, result=results)
    
except Exception as e:
    return render_template_string(HTML_TEMPLATE, error=str(e))


@app.route(’/api/analyze-multi’, methods=[‘POST’])
def api_analyze_multi():
“”“API endpoint for multi-timeframe analysis”””
try:
min_touches = int(request.form.get(‘min_touches’, 4))
atr_tolerance = float(request.form.get(‘atr_tolerance’, 0.1))


    timeframe_data = {}
    
    # Process files
    if 'file_1d' not in request.files:
        return jsonify({'error': '1D file is required'}), 400
    
    # Process 1D (required)
    file_1d = request.files['file_1d']
    csv_data = file_1d.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_data))
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    timeframe_data['1D'] = df
    
    # Process optional files
    for tf_key, file_key in [('4H', 'file_4h'), ('1H', 'file_1h')]:
        if file_key in request.files and request.files[file_key].filename != '':
            try:
                file_data = request.files[file_key].read().decode('utf-8')
                df_tf = pd.read_csv(io.StringIO(file_data))
                if 'Date' in df_tf.columns:
                    df_tf['Date'] = pd.to_datetime(df_tf['Date'])
                    df_tf.set_index('Date', inplace=True)
                timeframe_data[tf_key] = df_tf
            except Exception as e:
                print(f"Warning: Could not process {tf_key} file: {e}")
    
    finder = MultiTimeframeSRFinder(timeframe_data, min_touches, atr_tolerance)
    results = finder.get_detailed_results()
    
    return jsonify({
        'success': True,
        'levels': results['levels_csv'],
        'count': results['total_count'],
        'timeframes_used': results['timeframes_used']
    })
    
except Exception as e:
    return jsonify({'error': str(e)}), 400


@app.route(’/health’)
def health():
return jsonify({‘status’: ‘healthy’})

if **name** == ‘**main**’:
port = int(os.environ.get(‘PORT’, 5000))
app.run(host=‘0.0.0.0’, port=port, debug=True)
