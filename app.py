from flask import Flask, request, render_template_string, jsonify, session
import pandas as pd
import numpy as np
import io
import os

app = Flask(__name__)
app.secret_key = 'sr-levels-secret-key-2024'

class MultiTimeframeSRFinder:
    def __init__(self, timeframe_data, min_touches=2, tolerance_percentage=0.01, grouping_method="conservative", tolerance_mode="current_price"):
        self.timeframe_data = timeframe_data
        self.min_touches = min_touches
        self.tolerance_percentage = tolerance_percentage / 100.0
        self.grouping_method = grouping_method
        self.tolerance_mode = tolerance_mode
        self.timeframe_weights = {'1D': 3, '4H': 2, '1H': 1}
        self.prepare_data()
        
    def prepare_data(self):
        for timeframe, df in self.timeframe_data.items():
            df_copy = df.copy()
            
            # Handle column mapping
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
            
            if column_mapping:
                df_copy.rename(columns=column_mapping, inplace=True)
            
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df_copy.columns]
            if missing_cols:
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
        
        primary_timeframe = list(self.timeframe_data.keys())[0]
        self.current_price = self.timeframe_data[primary_timeframe]['Close'].iloc[-1]
        self.base_tolerance = self.current_price * self.tolerance_percentage
        
        print(f"Current price: ${self.current_price:.2f}")
        print(f"Base tolerance: ${self.base_tolerance:.3f}")
        print(f"Tolerance mode: {self.tolerance_mode}")
    
    def get_tolerance_for_price(self, price):
        if self.tolerance_mode == "current_price":
            return self.base_tolerance
        else:
            return price * self.tolerance_percentage
    
    def group_prices_conservative(self, prices, level_type, timeframe, weight):
        if not prices or len(prices) == 0:
            return []
        
        try:
            sorted_prices = sorted([p for p in prices if p is not None and not pd.isna(p)])
            if not sorted_prices:
                return []
                
            groups = []
            current_group = [sorted_prices[0]]
            
            for price in sorted_prices[1:]:
                if self.tolerance_mode == "level_price":
                    group_avg = sum(current_group) / len(current_group)
                    tolerance = self.get_tolerance_for_price(group_avg)
                else:
                    tolerance = self.base_tolerance
                
                min_distance = min(abs(price - group_price) for group_price in current_group)
                
                if min_distance <= tolerance:
                    current_group.append(price)
                else:
                    if len(current_group) >= 1:
                        groups.append(current_group.copy())
                    current_group = [price]
            
            if len(current_group) >= 1:
                groups.append(current_group)
            
            return self.convert_groups_to_levels(groups, level_type, timeframe, weight)
        
        except Exception as e:
            print(f"Error in conservative grouping: {e}")
            return []
    
    def group_prices_aggressive(self, prices, level_type, timeframe, weight):
        if not prices or len(prices) == 0:
            return []
        
        try:
            sorted_prices = sorted([p for p in prices if p is not None and not pd.isna(p)])
            if not sorted_prices:
                return []
                
            groups = []
            current_group = [sorted_prices[0]]
            
            for price in sorted_prices[1:]:
                group_center = sum(current_group) / len(current_group)
                
                if self.tolerance_mode == "level_price":
                    tolerance = self.get_tolerance_for_price(group_center)
                else:
                    tolerance = self.base_tolerance
                
                if abs(price - group_center) <= tolerance:
                    current_group.append(price)
                else:
                    if len(current_group) >= 1:
                        groups.append(current_group.copy())
                    current_group = [price]
            
            if len(current_group) >= 1:
                groups.append(current_group)
            
            return self.convert_groups_to_levels(groups, level_type, timeframe, weight)
        
        except Exception as e:
            print(f"Error in aggressive grouping: {e}")
            return []
    
    def convert_groups_to_levels(self, groups, level_type, timeframe, weight):
        levels = []
        for group in groups:
            if len(group) >= 1:
                if len(group) >= 3:
                    level_price = np.median(group)
                else:
                    level_price = sum(group) / len(group)
                
                tolerance_used = self.get_tolerance_for_price(level_price)
                
                levels.append({
                    'level': level_price,
                    'type': level_type,
                    'touches': len(group),
                    'timeframe': timeframe,
                    'weight': weight,
                    'weighted_touches': len(group) * weight,
                    'original_prices': group,
                    'price_range': f"${min(group):.2f}-${max(group):.2f}" if len(group) > 1 else f"${group[0]:.2f}",
                    'tolerance_used': tolerance_used,
                    'price_spread': max(group) - min(group) if len(group) > 1 else 0
                })
        
        return levels
    
    def find_levels_for_timeframe(self, timeframe, df):
        weight = self.timeframe_weights.get(timeframe, 1)
        
        if df is None or df.empty:
            print(f"Warning: No data for timeframe {timeframe}")
            return []
        
        try:
            high_prices = df['High'].dropna().tolist()
            low_prices = df['Low'].dropna().tolist()
        except KeyError as e:
            print(f"Error: Missing column in {timeframe}: {e}")
            return []
        
        if not high_prices or not low_prices:
            print(f"Warning: No price data found for {timeframe}")
            return []
        
        try:
            if self.grouping_method == "conservative":
                resistance_levels = self.group_prices_conservative(high_prices, 'Resistance', timeframe, weight)
                support_levels = self.group_prices_conservative(low_prices, 'Support', timeframe, weight)
            else:
                resistance_levels = self.group_prices_aggressive(high_prices, 'Resistance', timeframe, weight)
                support_levels = self.group_prices_aggressive(low_prices, 'Support', timeframe, weight)
        except Exception as e:
            print(f"Error in grouping for {timeframe}: {e}")
            return []
        
        all_levels = (resistance_levels or []) + (support_levels or [])
        return all_levels
    
    def combine_multi_timeframe_levels(self):
        all_levels = []
        
        for timeframe, df in self.timeframe_data.items():
            tf_levels = self.find_levels_for_timeframe(timeframe, df)
            all_levels.extend(tf_levels)
        
        grouped_levels = self.group_similar_levels(all_levels)
        
        strong_levels = []
        for group in grouped_levels:
            total_touches = sum(level['touches'] for level in group)
            total_weighted_touches = sum(level['weighted_touches'] for level in group)
            
            if total_touches >= self.min_touches:
                weighted_sum = sum(level['level'] * level['weighted_touches'] for level in group)
                final_level = weighted_sum / total_weighted_touches
                
                types = [level['type'] for level in group]
                final_type = max(set(types), key=types.count)
                
                timeframes = list(set(level['timeframe'] for level in group))
                
                tolerance_info = {
                    'tolerance_used': self.get_tolerance_for_price(final_level),
                    'tolerance_mode': self.tolerance_mode,
                    'tolerance_percentage': self.tolerance_percentage * 100
                }
                
                strong_levels.append({
                    'level': final_level,
                    'type': final_type,
                    'touches': total_touches,
                    'weighted_touches': total_weighted_touches,
                    'timeframes': timeframes,
                    'timeframe_count': len(timeframes),
                    'tolerance_info': tolerance_info,
                    'source_levels': len(group)
                })
        
        strong_levels.sort(key=lambda x: (x['timeframe_count'], x['weighted_touches']), reverse=True)
        return strong_levels
    
    def group_similar_levels(self, all_levels):
        if not all_levels:
            return []
        
        sorted_levels = sorted(all_levels, key=lambda x: x['level'])
        groups = []
        current_group = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            tolerance = self.get_tolerance_for_price(level['level'])
            
            if any(abs(level['level'] - group_level['level']) <= tolerance 
                   for group_level in current_group):
                current_group.append(level)
            else:
                groups.append(current_group)
                current_group = [level]
        
        groups.append(current_group)
        return groups
    
    def analyze_missing_levels(self, price_range_start, price_range_end):
        analysis = {
            'range': f"${price_range_start}-${price_range_end}",
            'current_tolerance': self.base_tolerance,
            'prices_in_range': []
        }
        
        for timeframe, df in self.timeframe_data.items():
            for price_type in ['High', 'Low']:
                prices_in_range = df[df[price_type].between(price_range_start, price_range_end)][price_type].tolist()
                for price in prices_in_range:
                    analysis['prices_in_range'].append({
                        'timeframe': timeframe,
                        'type': price_type,
                        'price': price
                    })
        
        if analysis['prices_in_range']:
            prices = [p['price'] for p in analysis['prices_in_range']]
            unique_prices = sorted(list(set(prices)))
            
            if len(unique_prices) > 1:
                min_distance = min(abs(unique_prices[i+1] - unique_prices[i]) 
                                 for i in range(len(unique_prices)-1))
                mid_range_price = (price_range_start + price_range_end) / 2
                suggested_tolerance_pct = (min_distance / mid_range_price) * 100
                
                analysis['suggested_tolerance_percentage'] = suggested_tolerance_pct
                analysis['min_distance'] = min_distance
                analysis['unique_price_count'] = len(unique_prices)
        
        return analysis
    
    def get_detailed_results(self):
        levels = self.combine_multi_timeframe_levels()
        level_prices = [f"{level['level']:.2f}" for level in levels]
        
        return {
            'levels_csv': ",".join(level_prices),
            'total_count': len(levels),
            'detailed_levels': levels,
            'timeframes_used': list(self.timeframe_data.keys()),
            'tolerance_info': {
                'percentage': self.tolerance_percentage * 100,
                'dollar_amount': self.base_tolerance,
                'current_price': self.current_price,
                'tolerance_mode': self.tolerance_mode
            },
            'grouping_method': self.grouping_method,
            'min_touches': self.min_touches
        }

# Simple, reliable HTML template with dropdowns
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Timeframe S&R Level Finder</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 900px; 
            margin: 20px auto; 
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
            margin-bottom: 15px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
            font-style: italic;
        }
        .form-group { 
            margin-bottom: 15px; 
        }
        .file-group {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .file-group label {
            width: 60px;
            font-weight: bold;
            margin-right: 15px;
        }
        .file-status {
            margin-left: 10px;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
        }
        .file-loaded {
            background: #d4edda;
            color: #155724;
        }
        .file-required {
            background: #f8d7da;
            color: #721c24;
        }
        label { 
            display: block; 
            margin-bottom: 5px; 
            font-weight: bold;
        }
        input, select { 
            width: 100%; 
            padding: 8px; 
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
            margin-top: 15px;
        }
        button:hover { 
            background: #0056b3; 
        }
        .btn-secondary {
            background: #6c757d;
            width: auto;
            margin: 5px;
        }
        .btn-warning {
            background: #ffc107;
            color: #212529;
        }
        .result { 
            margin-top: 25px; 
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
        .section {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            background: #fafafa;
        }
        .settings-section {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
        }
        .info {
            background: #d1ecf1;
            color: #0c5460;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .tolerance-warning {
            background: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .diagnostic-info {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Multi-Timeframe S&R Level Finder</h1>
        <div class="subtitle">Upload multiple timeframes • Level-based tolerance • File persistence</div>
        
        <div class="info">
            <strong>🔍 Enhanced Features:</strong><br>
            • Multi-timeframe analysis (1D, 4H, 1H)<br>
            • Files persist between tests (no re-upload needed)<br>
            • Level-based tolerance (fixes missing levels issue)<br>
            • Range analysis for missing levels<br>
            • Works universally (penny stocks to Berkshire!)
        </div>

        <form method="post" enctype="multipart/form-data">
            <div class="section">
                <h3>📁 Upload Timeframe Data</h3>
                <div class="file-group">
                    <label for="file_1d">1D:</label>
                    <input type="file" name="file_1d" accept=".csv">
                    <span class="file-status {{ 'file-loaded' if files_loaded.get('1D') else 'file-required' }}">
                        {{ 'Loaded ✓' if files_loaded.get('1D') else 'Required' }}
                    </span>
                </div>
                
                <div class="file-group">
                    <label for="file_4h">4H:</label>
                    <input type="file" name="file_4h" accept=".csv">
                    <span class="file-status {{ 'file-loaded' if files_loaded.get('4H') else '' }}">
                        {{ 'Loaded ✓' if files_loaded.get('4H') else 'Optional' }}
                    </span>
                </div>
                
                <div class="file-group">
                    <label for="file_1h">1H:</label>
                    <input type="file" name="file_1h" accept=".csv">
                    <span class="file-status {{ 'file-loaded' if files_loaded.get('1H') else '' }}">
                        {{ 'Loaded ✓' if files_loaded.get('1H') else 'Optional' }}
                    </span>
                </div>
                
                <button type="submit" name="action" value="clear_files" class="btn-secondary btn-warning">Clear All Files</button>
            </div>

            <div class="section settings-section">
                <h3>⚙️ Analysis Settings</h3>
                
                <div class="form-group">
                    <label for="tolerance_mode">🎯 Tolerance Calculation Mode:</label>
                    <select name="tolerance_mode">
                        <option value="current_price" {{ 'selected' if last_settings.get('tolerance_mode') == 'current_price' else '' }}>
                            Current Price Based (Original) - May miss mid-range levels
                        </option>
                        <option value="level_price" {{ 'selected' if last_settings.get('tolerance_mode') == 'level_price' else '' }}>
                            Level Price Based (NEW) - ⭐ RECOMMENDED for all stocks
                        </option>
                    </select>
                    <div style="font-size: 12px; color: #666; margin-top: 5px;">
                        <strong>Current Price:</strong> 0.01% of $176 = $0.018 tolerance everywhere<br>
                        <strong>Level Price:</strong> 0.01% of $155 = $0.016 tolerance at $155 level (adaptive)
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="grouping_method">📊 Grouping Method:</label>
                    <select name="grouping_method">
                        <option value="conservative" {{ 'selected' if last_settings.get('grouping_method') == 'conservative' else '' }}>
                            Conservative - Preserves distinct levels (recommended)
                        </option>
                        <option value="aggressive" {{ 'selected' if last_settings.get('grouping_method') == 'aggressive' else '' }}>
                            Aggressive - Groups similar levels more
                        </option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="tolerance_percentage">📏 Tolerance (% of reference price):</label>
                    <select name="tolerance_percentage">
                        <option value="0.005" {{ 'selected' if last_settings.get('tolerance_percentage') == '0.005' else '' }}>0.005% - Ultra Tight</option>
                        <option value="0.01" {{ 'selected' if last_settings.get('tolerance_percentage') == '0.01' else '' }}>0.01% - Tight (default)</option>
                        <option value="0.015" {{ 'selected' if last_settings.get('tolerance_percentage') == '0.015' else '' }}>0.015% - Moderate</option>
                        <option value="0.02" {{ 'selected' if last_settings.get('tolerance_percentage') == '0.02' else '' }}>0.02% - Balanced ⭐ (NVDA recommended)</option>
                        <option value="0.03" {{ 'selected' if last_settings.get('tolerance_percentage') == '0.03' else '' }}>0.03% - Loose</option>
                        <option value="0.05" {{ 'selected' if last_settings.get('tolerance_percentage') == '0.05' else '' }}>0.05% - Very Loose</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="min_touches">🎯 Minimum Total Touches:</label>
                    <select name="min_touches">
                        <option value="1" {{ 'selected' if last_settings.get('min_touches') == '1' else '' }}>1 - Maximum Sensitivity</option>
                        <option value="2" {{ 'selected' if last_settings.get('min_touches') == '2' else '' }}>2 - Very Sensitive (default)</option>
                        <option value="3" {{ 'selected' if last_settings.get('min_touches') == '3' else '' }}>3 - Sensitive</option>
                        <option value="4" {{ 'selected' if last_settings.get('min_touches') == '4' else '' }}>4 - Balanced</option>
                        <option value="5" {{ 'selected' if last_settings.get('min_touches') == '5' else '' }}>5 - Conservative</option>
                        <option value="6" {{ 'selected' if last_settings.get('min_touches') == '6' else '' }}>6 - Very Conservative</option>
                    </select>
                </div>
            </div>

            <div class="section">
                <h3>🔍 Missing Range Analysis</h3>
                <div class="form-group">
                    <label for="analysis_range_start">Analyze Missing Levels From Price:</label>
                    <input type="number" name="analysis_range_start" step="0.01" 
                           value="{{ last_settings.get('analysis_range_start', '148') }}" 
                           placeholder="148.00">
                </div>
                <div class="form-group">
                    <label for="analysis_range_end">To Price:</label>
                    <input type="number" name="analysis_range_end" step="0.01" 
                           value="{{ last_settings.get('analysis_range_end', '170') }}" 
                           placeholder="170.00">
                </div>
            </div>
            
            <button type="submit" name="action" value="analyze">🎯 Find S&R Levels</button>
            
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <button type="submit" name="action" value="analyze_range" class="btn-secondary" style="flex: 1;">🔍 Analyze Missing Range Only</button>
            </div>
        </form>

        {% if result %}
        <div class="result">
            <h3>🎯 S&R Analysis Results:</h3>
            <p><strong>Files loaded:</strong> {{ result.timeframes_used | join(', ') }}</p>
            <p><strong>Total levels found:</strong> {{ result.total_count }}</p>
            <p><strong>Settings:</strong> {{ result.grouping_method.title() }} grouping, {{ result.min_touches }} min touches</p>
            <p><strong>Tolerance:</strong> {{ "%.3f"|format(result.tolerance_info.percentage) }}% {{ result.tolerance_info.tolerance_mode.replace('_', ' ') }} mode</p>
            <p><strong>Current price:</strong> ${{ "%.2f"|format(result.tolerance_info.current_price) }}</p>
            <p><strong>Base tolerance:</strong> ${{ "%.3f"|format(result.tolerance_info.dollar_amount) }}</p>
            
            <strong>📋 Levels for TradingView:</strong>
            <div class="levels-output">{{ result.levels_csv }}</div>
            
            <div style="margin-top: 15px;">
                <strong>📊 Level Details:</strong>
                <table>
                    <tr>
                        <th>Level</th>
                        <th>Type</th>
                        <th>Touches</th>
                        <th>Timeframes</th>
                        <th>Strength</th>
                        <th>Tolerance Used</th>
                    </tr>
                    {% for level in result.detailed_levels[:15] %}
                    <tr>
                        <td>${{ "%.2f"|format(level.level) }}</td>
                        <td>{{ level.type }}</td>
                        <td>{{ level.touches }}</td>
                        <td>{{ level.timeframes | join(', ') }}</td>
                        <td>{{ level.weighted_touches }}</td>
                        <td>${{ "%.3f"|format(level.tolerance_info.tolerance_used) if level.tolerance_info else 'N/A' }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% if result.detailed_levels|length > 15 %}
                <p><em>Showing top 15 levels...</em></p>
                {% endif %}
            </div>
        </div>
        {% endif %}

        {% if range_analysis %}
        <div class="result">
            <h3>🔍 Missing Range Analysis: {{ range_analysis.range }}</h3>
            <p><strong>Prices found in range:</strong> {{ range_analysis.prices_in_range | length }}</p>
            <p><strong>Current tolerance:</strong> ${{ "%.3f"|format(range_analysis.current_tolerance) }}</p>
            
            {% if range_analysis.get('suggested_tolerance_percentage') %}
            <div class="tolerance-warning">
                <strong>💡 Solution Found:</strong><br>
                To capture levels in this range, use <strong>{{ "%.3f"|format(range_analysis.suggested_tolerance_percentage) }}%</strong> tolerance<br>
                (Min distance between prices: ${{ "%.3f"|format(range_analysis.min_distance) }})<br>
                Found {{ range_analysis.unique_price_count }} unique price levels in range.
            </div>
            {% endif %}
            
            {% if range_analysis.prices_in_range %}
            <div class="diagnostic-info">
                <strong>Price occurrences in range (first 20):</strong><br>
                {% for price_info in range_analysis.prices_in_range[:20] %}
                {{ price_info.timeframe }} {{ price_info.type }}: ${{ "%.2f"|format(price_info.price) }}<br>
                {% endfor %}
                {% if range_analysis.prices_in_range | length > 20 %}
                ... and {{ range_analysis.prices_in_range | length - 20 }} more
                {% endif %}
            </div>
            {% endif %}
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

# File storage functions
def save_dataframe_to_session(df, key):
    """Save dataframe to session"""
    try:
        buffer = io.BytesIO()
        df.to_pickle(buffer)
        session[f'df_{key}'] = buffer.getvalue().hex()
        return True
    except Exception as e:
        print(f"Error saving dataframe: {e}")
        return False

def load_dataframe_from_session(key):
    """Load dataframe from session"""
    try:
        if f'df_{key}' in session:
            hex_data = session[f'df_{key}']
            buffer = io.BytesIO(bytes.fromhex(hex_data))
            return pd.read_pickle(buffer)
    except Exception as e:
        print(f"Error loading dataframe: {e}")
    return None

def process_uploaded_file(file_obj):
    """Process uploaded CSV file"""
    try:
        csv_data = file_obj.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Column mapping
        df.columns = df.columns.str.lower().str.strip()
        column_map = {
            'time': 'Date', 'date': 'Date', 'datetime': 'Date',
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'volume': 'Volume', 'vol': 'Volume',
            'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'
        }
        df.rename(columns=column_map, inplace=True)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        return df
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # Get session data for template
        files_loaded = session.get('files_loaded', {})
        last_settings = session.get('last_settings', {})
        
        # Render template with session data
        template = HTML_TEMPLATE
        template = template.replace('{{ files_loaded.get(\'1D\') }}', str(files_loaded.get('1D', False)).lower())
        template = template.replace('{{ files_loaded.get(\'4H\') }}', str(files_loaded.get('4H', False)).lower())
        template = template.replace('{{ files_loaded.get(\'1H\') }}', str(files_loaded.get('1H', False)).lower())
        
        # Replace settings
        template = template.replace('{{ last_settings.get(\'tolerance_mode\') }}', last_settings.get('tolerance_mode', ''))
        template = template.replace('{{ last_settings.get(\'grouping_method\') }}', last_settings.get('grouping_method', ''))
        template = template.replace('{{ last_settings.get(\'tolerance_percentage\') }}', last_settings.get('tolerance_percentage', ''))
        template = template.replace('{{ last_settings.get(\'min_touches\') }}', last_settings.get('min_touches', ''))
        template = template.replace('{{ last_settings.get(\'analysis_range_start\', \'148\') }}', last_settings.get('analysis_range_start', '148'))
        template = template.replace('{{ last_settings.get(\'analysis_range_end\', \'170\') }}', last_settings.get('analysis_range_end', '170'))
        
        # Clean up template conditions
        template = template.replace('{{ \'file-loaded\' if files_loaded.get(\'1D\') else \'file-required\' }}', 'file-loaded' if files_loaded.get('1D') else 'file-required')
        template = template.replace('{{ \'file-loaded\' if files_loaded.get(\'4H\') else \'\' }}', 'file-loaded' if files_loaded.get('4H') else '')
        template = template.replace('{{ \'file-loaded\' if files_loaded.get(\'1H\') else \'\' }}', 'file-loaded' if files_loaded.get('1H') else '')
        
        template = template.replace('{{ \'Loaded ✓\' if files_loaded.get(\'1D\') else \'Required\' }}', 'Loaded ✓' if files_loaded.get('1D') else 'Required')
        template = template.replace('{{ \'Loaded ✓\' if files_loaded.get(\'4H\') else \'Optional\' }}', 'Loaded ✓' if files_loaded.get('4H') else 'Optional')
        template = template.replace('{{ \'Loaded ✓\' if files_loaded.get(\'1H\') else \'Optional\' }}', 'Loaded ✓' if files_loaded.get('1H') else 'Optional')
        
        # Handle selected attributes
        tolerance_mode = last_settings.get('tolerance_mode', '')
        template = template.replace('{{ \'selected\' if last_settings.get(\'tolerance_mode\') == \'current_price\' else \'\' }}', 'selected' if tolerance_mode == 'current_price' else '')
        template = template.replace('{{ \'selected\' if last_settings.get(\'tolerance_mode\') == \'level_price\' else \'\' }}', 'selected' if tolerance_mode == 'level_price' else '')
        
        grouping_method = last_settings.get('grouping_method', '')
        template = template.replace('{{ \'selected\' if last_settings.get(\'grouping_method\') == \'conservative\' else \'\' }}', 'selected' if grouping_method == 'conservative' else '')
        template = template.replace('{{ \'selected\' if last_settings.get(\'grouping_method\') == \'aggressive\' else \'\' }}', 'selected' if grouping_method == 'aggressive' else '')
        
        tolerance_pct = last_settings.get('tolerance_percentage', '')
        for pct in ['0.005', '0.01', '0.015', '0.02', '0.03', '0.05']:
            template = template.replace(f'{{{{ \'selected\' if last_settings.get(\'tolerance_percentage\') == \'{pct}\' else \'\' }}}}', 'selected' if tolerance_pct == pct else '')
        
        min_touches = last_settings.get('min_touches', '')
        for touches in ['1', '2', '3', '4', '5', '6']:
            template = template.replace(f'{{{{ \'selected\' if last_settings.get(\'min_touches\') == \'{touches}\' else \'\' }}}}', 'selected' if min_touches == touches else '')
        
        # Remove any remaining template syntax
        template = template.replace('{% if result %}', '<!-- ').replace('{% if range_analysis %}', '<!-- ').replace('{% if error %}', '<!-- ').replace('{% endif %}', ' -->')
        
        return template
    
    try:
        action = request.form.get('action', 'analyze')
        
        # Handle clear files action
        if action == 'clear_files':
            session.clear()
            return index()  # Reload the page
        
        # Get settings and save to session
        settings = {
            'min_touches': int(request.form.get('min_touches', 2)),
            'tolerance_percentage': request.form.get('tolerance_percentage', '0.01'),
            'grouping_method': request.form.get('grouping_method', 'conservative'),
            'tolerance_mode': request.form.get('tolerance_mode', 'current_price'),
            'analysis_range_start': request.form.get('analysis_range_start', '148'),
            'analysis_range_end': request.form.get('analysis_range_end', '170')
        }
        session['last_settings'] = settings
        
        # Process files
        timeframe_data = {}
        files_loaded = session.get('files_loaded', {})
        
        for tf_key, file_key in [('1D', 'file_1d'), ('4H', 'file_4h'), ('1H', 'file_1h')]:
            # New file uploaded
            if file_key in request.files and request.files[file_key].filename != '':
                try:
                    df = process_uploaded_file(request.files[file_key])
                    if save_dataframe_to_session(df, tf_key):
                        files_loaded[tf_key] = True
                        timeframe_data[tf_key] = df
                except Exception as e:
                    return render_template_string(HTML_TEMPLATE, error=f"Error processing {tf_key} file: {str(e)}")
            # Use cached data
            elif tf_key in files_loaded:
                df = load_dataframe_from_session(tf_key)
                if df is not None:
                    timeframe_data[tf_key] = df
        
        session['files_loaded'] = files_loaded
        
        # Validate
        if '1D' not in timeframe_data:
            return render_template_string(HTML_TEMPLATE, error="1D timeframe file is required")
        
        # Handle range analysis only
        if action == 'analyze_range':
            finder = MultiTimeframeSRFinder(
                timeframe_data, 
                settings['min_touches'], 
                float(settings['tolerance_percentage']), 
                settings['grouping_method'],
                settings['tolerance_mode']
            )
            
            range_start = float(settings['analysis_range_start'])
            range_end = float(settings['analysis_range_end'])
            range_analysis = finder.analyze_missing_levels(range_start, range_end)
            
            return render_template_string(HTML_TEMPLATE, range_analysis=range_analysis, 
                                        files_loaded=files_loaded, last_settings=settings)
        
        # Full analysis
        finder = MultiTimeframeSRFinder(
            timeframe_data, 
            settings['min_touches'], 
            float(settings['tolerance_percentage']), 
            settings['grouping_method'],
            settings['tolerance_mode']
        )
        
        results = finder.get_detailed_results()
        
        # Range analysis
        range_analysis = None
        if settings['analysis_range_start'] and settings['analysis_range_end']:
            try:
                range_start = float(settings['analysis_range_start'])
                range_end = float(settings['analysis_range_end'])
                range_analysis = finder.analyze_missing_levels(range_start, range_end)
            except:
                pass
        
        return render_template_string(HTML_TEMPLATE, result=results, range_analysis=range_analysis,
                                    files_loaded=files_loaded, last_settings=settings)
        
    except Exception as e:
        files_loaded = session.get('files_loaded', {})
        last_settings = session.get('last_settings', {})
        return render_template_string(HTML_TEMPLATE, error=str(e), 
                                    files_loaded=files_loaded, last_settings=last_settings)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'Multi-Timeframe S&R App is running'})

@app.route('/test')
def test():
    return '<h1>Multi-Timeframe S&R App is Working!</h1><p>Go to <a href="/">/</a> to upload files and find levels.</p>'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Production-ready configuration
    app.run(host='0.0.0.0', port=port, debug=False)
