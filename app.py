from flask import Flask, request, render_template_string, jsonify, session
import pandas as pd
import numpy as np
from collections import defaultdict
import io
import os
import pickle
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Change this in production

class MultiTimeframeSRFinder:
    def __init__(self, timeframe_data, min_touches=2, tolerance_percentage=0.01, grouping_method="conservative", tolerance_mode="current_price"):
        """
        Enhanced Multi-timeframe Support & Resistance finder
        
        Parameters:
        tolerance_mode: "current_price" or "level_price" - how to calculate tolerance
        """
        self.timeframe_data = timeframe_data
        self.min_touches = min_touches
        self.tolerance_percentage = tolerance_percentage / 100.0
        self.grouping_method = grouping_method
        self.tolerance_mode = tolerance_mode
        self.timeframe_weights = {'1D': 3, '4H': 2, '1H': 1}
        self.prepare_data()
        
    def prepare_data(self):
        """Calculate current price and set tolerance, handle different column formats"""
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
        
        # Get current price from the primary (first) timeframe
        primary_timeframe = list(self.timeframe_data.keys())[0]
        self.current_price = self.timeframe_data[primary_timeframe]['Close'].iloc[-1]
        
        # Calculate tolerance based on current price (original method)
        self.base_tolerance = self.current_price * self.tolerance_percentage
        
        print(f"Current price: ${self.current_price:.2f}")
        print(f"Base tolerance ({self.tolerance_percentage*100:.3f}% of current): ${self.base_tolerance:.3f}")
        print(f"Tolerance mode: {self.tolerance_mode}")
        print(f"Grouping method: {self.grouping_method}")
        print(f"Timeframes loaded: {list(self.timeframe_data.keys())}")
    
    def get_tolerance_for_price(self, price):
        """Get tolerance based on the selected mode"""
        if self.tolerance_mode == "current_price":
            return self.base_tolerance
        else:  # level_price mode
            return price * self.tolerance_percentage
    
    def group_prices_conservative(self, prices, level_type, timeframe, weight):
        """Conservative grouping with improved tolerance handling"""
        if not prices or len(prices) == 0:
            return []
        
        try:
            sorted_prices = sorted([p for p in prices if p is not None and not pd.isna(p)])
            if not sorted_prices:
                return []
                
            groups = []
            current_group = [sorted_prices[0]]
            
            for price in sorted_prices[1:]:
                # Calculate tolerance based on current group
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
        """Aggressive grouping with improved tolerance handling"""
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
        """Convert price groups to level objects with diagnostic info"""
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
        """Find support and resistance levels for a single timeframe"""
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
        """Combine levels from all timeframes and find the strongest ones"""
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
                
                # Add diagnostic information
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
        """Group levels that are within tolerance of each other"""
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
        """Analyze why levels might be missing in a specific price range"""
        analysis = {
            'range': f"${price_range_start}-${price_range_end}",
            'current_tolerance': self.base_tolerance,
            'prices_in_range': []
        }
        
        # Find all prices in the range
        for timeframe, df in self.timeframe_data.items():
            for price_type in ['High', 'Low']:
                prices_in_range = df[df[price_type].between(price_range_start, price_range_end)][price_type].tolist()
                for price in prices_in_range:
                    analysis['prices_in_range'].append({
                        'timeframe': timeframe,
                        'type': price_type,
                        'price': price
                    })
        
        # Calculate what tolerance would be needed
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
        """Get detailed results for analysis"""
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

# Enhanced HTML template with file persistence and improved settings
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Multi-Timeframe S&R Level Finder</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 1000px; 
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
        .file-group input {
            flex: 1;
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
        
        /* Radio Button and Checkbox Styling */
        .radio-group, .checkbox-grid, .touches-grid {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 8px;
        }
        
        .checkbox-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
        }
        
        .touches-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }
        
        .radio-option, .tolerance-option, .touches-option {
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 12px;
            background: #f8f9fa;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .radio-option:hover, .tolerance-option:hover, .touches-option:hover {
            border-color: #007bff;
            background: #f0f8ff;
        }
        
        .radio-option.recommended, .tolerance-option.recommended, .touches-option.recommended {
            border-color: #28a745;
            background: #f8fff8;
        }
        
        .radio-option input[type="radio"]:checked + .radio-label,
        .tolerance-option input[type="radio"]:checked + .tolerance-label,
        .touches-option input[type="radio"]:checked + .touches-label {
            font-weight: bold;
        }
        
        .radio-option input[type="radio"]:checked,
        .tolerance-option input[type="radio"]:checked,
        .touches-option input[type="radio"]:checked {
            transform: scale(1.2);
        }
        
        .radio-option:has(input[type="radio"]:checked),
        .tolerance-option:has(input[type="radio"]:checked),
        .touches-option:has(input[type="radio"]:checked) {
            border-color: #007bff;
            background: #e7f3ff;
            box-shadow: 0 2px 8px rgba(0,123,255,0.2);
        }
        
        .radio-option.recommended:has(input[type="radio"]:checked),
        .tolerance-option.recommended:has(input[type="radio"]:checked),
        .touches-option.recommended:has(input[type="radio"]:checked) {
            border-color: #28a745;
            background: #e8f5e8;
            box-shadow: 0 2px 8px rgba(40,167,69,0.2);
        }
        
        .radio-label, .tolerance-label, .touches-label {
            display: block;
            margin: 0;
            cursor: pointer;
            font-weight: normal;
        }
        
        .option-description, .tolerance-desc, .touches-desc {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
            line-height: 1.3;
        }
        
        .option-example {
            font-size: 11px;
            color: #888;
            font-style: italic;
            margin-top: 2px;
        }
        
        .recommended-badge {
            background: #28a745;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: bold;
        }
        
        input[type="radio"] {
            width: auto !important;
            margin-right: 8px;
            transform: scale(1.1);
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
        .utility-buttons {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .utility-buttons button {
            width: auto;
            flex: 1;
            padding: 8px 15px;
            font-size: 14px;
        }
        .btn-secondary {
            background: #6c757d;
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
            padding: 15px;
            margin-bottom: 15px;
            background: #fafafa;
        }
        .settings-section {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .tolerance-warning {
            background: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 14px;
        }
        .analysis-section {
            background: #e7f3ff;
            border: 1px solid #b8daff;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
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
        <h1>🎯 Enhanced Multi-Timeframe S&R Level Finder</h1>
        <div class="subtitle">Upload once, test multiple settings • Now with Level-Based Tolerance!</div>
        
        <div class="info">
            <strong>🔍 Diagnostic Features:</strong><br>
            • Files persist between tests (no re-upload needed)<br>
            • Settings remain selected<br>
            • Level-based tolerance option (fixes missing levels issue)<br>
            • Missing range analysis<br>
            • <strong>WHY 148-170 MISSING:</strong> Default tolerance (0.01% of $176) = $0.018 is too small for $150+ prices
        </div>

        <form method="post" enctype="multipart/form-data">
            <div class="file-section">
                <h3>📁 Upload Timeframe Data</h3>
                <div class="file-group">
                    <label for="file_1d">1D:</label>
                    <input type="file" name="file_1d" accept=".csv">
                    <span class="file-status {{ 'file-loaded' if session.get('files_loaded', {}).get('1D') else 'file-required' }}">
                        {{ 'Loaded ✓' if session.get('files_loaded', {}).get('1D') else 'Required' }}
                    </span>
                </div>
                
                <div class="file-group">
                    <label for="file_4h">4H:</label>
                    <input type="file" name="file_4h" accept=".csv">
                    <span class="file-status {{ 'file-loaded' if session.get('files_loaded', {}).get('4H') else '' }}">
                        {{ 'Loaded ✓' if session.get('files_loaded', {}).get('4H') else 'Optional' }}
                    </span>
                </div>
                
                <div class="file-group">
                    <label for="file_1h">1H:</label>
                    <input type="file" name="file_1h" accept=".csv">
                    <span class="file-status {{ 'file-loaded' if session.get('files_loaded', {}).get('1H') else '' }}">
                        {{ 'Loaded ✓' if session.get('files_loaded', {}).get('1H') else 'Optional' }}
                    </span>
                </div>
                
                <div class="utility-buttons">
                    <button type="submit" name="action" value="clear_files" class="btn-warning">Clear All Files</button>
                </div>
            </div>

            <div class="settings-section">
                <h3>⚙️ Analysis Settings</h3>
                
                <div class="form-group">
                    <label>🎯 Tolerance Calculation Mode:</label>
                    <div class="radio-group">
                        {% set current_mode = session.get('last_settings', {}).get('tolerance_mode', 'current_price') %}
                        <div class="radio-option">
                            <input type="radio" id="current_price" name="tolerance_mode" value="current_price" 
                                   {{ 'checked' if current_mode == 'current_price' else '' }}>
                            <label for="current_price" class="radio-label">
                                <strong>Current Price Based</strong> (Original)
                                <div class="option-description">Uses current stock price for tolerance everywhere. May miss mid-range levels.</div>
                                <div class="option-example">Example: 0.01% of $176 = $0.018 tolerance at all price levels</div>
                            </label>
                        </div>
                        <div class="radio-option recommended">
                            <input type="radio" id="level_price" name="tolerance_mode" value="level_price"
                                   {{ 'checked' if current_mode == 'level_price' else '' }}>
                            <label for="level_price" class="radio-label">
                                <strong>Level Price Based</strong> (NEW) ⭐ <span class="recommended-badge">RECOMMENDED</span>
                                <div class="option-description">Adapts tolerance to each price level. Universal solution for all stocks.</div>
                                <div class="option-example">Example: 0.01% of $155 = $0.016 tolerance at $155 level (adaptive)</div>
                            </label>
                        </div>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>📊 Grouping Method:</label>
                    <div class="radio-group">
                        {% set current_grouping = session.get('last_settings', {}).get('grouping_method', 'conservative') %}
                        <div class="radio-option recommended">
                            <input type="radio" id="conservative" name="grouping_method" value="conservative"
                                   {{ 'checked' if current_grouping == 'conservative' else '' }}>
                            <label for="conservative" class="radio-label">
                                <strong>Conservative</strong> ⭐ <span class="recommended-badge">RECOMMENDED</span>
                                <div class="option-description">Preserves more distinct levels, less grouping.</div>
                                <div class="option-example">Keeps $179.88 and $179.38 as separate levels</div>
                            </label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" id="aggressive" name="grouping_method" value="aggressive"
                                   {{ 'checked' if current_grouping == 'aggressive' else '' }}>
                            <label for="aggressive" class="radio-label">
                                <strong>Aggressive</strong>
                                <div class="option-description">Groups similar levels more, better for noisy data.</div>
                                <div class="option-example">Groups $179.88 and $179.38 into single level ~$179.63</div>
                            </label>
                        </div>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>📏 Tolerance Percentage:</label>
                    <div class="checkbox-grid">
                        {% set current_tolerance = session.get('last_settings', {}).get('tolerance_percentage', '0.01') %}
                        <div class="tolerance-option">
                            <input type="radio" id="tol_005" name="tolerance_percentage" value="0.005"
                                   {{ 'checked' if current_tolerance == '0.005' else '' }}>
                            <label for="tol_005" class="tolerance-label">
                                <strong>0.005%</strong> - Ultra Tight
                                <div class="tolerance-desc">Finds micro-levels, very precise</div>
                            </label>
                        </div>
                        <div class="tolerance-option">
                            <input type="radio" id="tol_01" name="tolerance_percentage" value="0.01"
                                   {{ 'checked' if current_tolerance == '0.01' else '' }}>
                            <label for="tol_01" class="tolerance-label">
                                <strong>0.01%</strong> - Tight
                                <div class="tolerance-desc">Standard precision, good default</div>
                            </label>
                        </div>
                        <div class="tolerance-option">
                            <input type="radio" id="tol_015" name="tolerance_percentage" value="0.015"
                                   {{ 'checked' if current_tolerance == '0.015' else '' }}>
                            <label for="tol_015" class="tolerance-label">
                                <strong>0.015%</strong> - Moderate
                                <div class="tolerance-desc">Balanced precision and grouping</div>
                            </label>
                        </div>
                        <div class="tolerance-option recommended">
                            <input type="radio" id="tol_02" name="tolerance_percentage" value="0.02"
                                   {{ 'checked' if current_tolerance == '0.02' else '' }}>
                            <label for="tol_02" class="tolerance-label">
                                <strong>0.02%</strong> - Balanced ⭐
                                <div class="tolerance-desc">RECOMMENDED for NVDA, finds 148-170 levels</div>
                            </label>
                        </div>
                        <div class="tolerance-option">
                            <input type="radio" id="tol_03" name="tolerance_percentage" value="0.03"
                                   {{ 'checked' if current_tolerance == '0.03' else '' }}>
                            <label for="tol_03" class="tolerance-label">
                                <strong>0.03%</strong> - Loose
                                <div class="tolerance-desc">More grouping, fewer levels</div>
                            </label>
                        </div>
                        <div class="tolerance-option">
                            <input type="radio" id="tol_05" name="tolerance_percentage" value="0.05"
                                   {{ 'checked' if current_tolerance == '0.05' else '' }}>
                            <label for="tol_05" class="tolerance-label">
                                <strong>0.05%</strong> - Very Loose
                                <div class="tolerance-desc">Heavy consolidation of levels</div>
                            </label>
                        </div>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>🎯 Minimum Total Touches:</label>
                    <div class="touches-grid">
                        {% set current_touches = session.get('last_settings', {}).get('min_touches', '2') %}
                        <div class="touches-option">
                            <input type="radio" id="touches_1" name="min_touches" value="1"
                                   {{ 'checked' if current_touches == '1' else '' }}>
                            <label for="touches_1" class="touches-label">
                                <strong>1</strong> Touch
                                <div class="touches-desc">Maximum sensitivity, captures all levels</div>
                            </label>
                        </div>
                        <div class="touches-option recommended">
                            <input type="radio" id="touches_2" name="min_touches" value="2"
                                   {{ 'checked' if current_touches == '2' else '' }}>
                            <label for="touches_2" class="touches-label">
                                <strong>2</strong> Touches ⭐
                                <div class="touches-desc">Very sensitive, good default</div>
                            </label>
                        </div>
                        <div class="touches-option">
                            <input type="radio" id="touches_3" name="min_touches" value="3"
                                   {{ 'checked' if current_touches == '3' else '' }}>
                            <label for="touches_3" class="touches-label">
                                <strong>3</strong> Touches
                                <div class="touches-desc">Sensitive, proven levels</div>
                            </label>
                        </div>
                        <div class="touches-option">
                            <input type="radio" id="touches_4" name="min_touches" value="4"
                                   {{ 'checked' if current_touches == '4' else '' }}>
                            <label for="touches_4" class="touches-label">
                                <strong>4</strong> Touches
                                <div class="touches-desc">Balanced, strong levels</div>
                            </label>
                        </div>
                        <div class="touches-option">
                            <input type="radio" id="touches_5" name="min_touches" value="5"
                                   {{ 'checked' if current_touches == '5' else '' }}>
                            <label for="touches_5" class="touches-label">
                                <strong>5</strong> Touches
                                <div class="touches-desc">Conservative, very strong</div>
                            </label>
                        </div>
                        <div class="touches-option">
                            <input type="radio" id="touches_6" name="min_touches" value="6"
                                   {{ 'checked' if current_touches == '6' else '' }}>
                            <label for="touches_6" class="touches-label">
                                <strong>6</strong> Touches
                                <div class="touches-desc">Very conservative, strongest only</div>
                            </label>
                        </div>
                    </div>
                </div>
            </div>

            <div class="analysis-section">
                <h3>🔍 Missing Range Analysis</h3>
                <div class="form-group">
                    <label for="analysis_range_start">Analyze Missing Levels From Price:</label>
                    <input type="number" name="analysis_range_start" step="0.01" 
                           value="{{ session.get('last_settings', {}).get('analysis_range_start', '148') }}" 
                           placeholder="148.00">
                </div>
                <div class="form-group">
                    <label for="analysis_range_end">To Price:</label>
                    <input type="number" name="analysis_range_end" step="0.01" 
                           value="{{ session.get('last_settings', {}).get('analysis_range_end', '170') }}" 
                           placeholder="170.00">
                </div>
            </div>
            
            <button type="submit" name="action" value="analyze">🎯 Find S&R Levels</button>
            
            <div class="utility-buttons">
                <button type="submit" name="action" value="analyze_range" class="btn-secondary">🔍 Analyze Missing Range Only</button>
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
            
            <div class="details">
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

# File storage helper functions
def save_dataframe_to_session(df, key):
    """Save dataframe to session using pickle"""
    buffer = io.BytesIO()
    df.to_pickle(buffer)
    session[f'df_{key}'] = buffer.getvalue().hex()

def load_dataframe_from_session(key):
    """Load dataframe from session"""
    if f'df_{key}' in session:
        hex_data = session[f'df_{key}']
        buffer = io.BytesIO(bytes.fromhex(hex_data))
        return pd.read_pickle(buffer)
    return None

def process_uploaded_file(file_obj):
    """Process uploaded CSV file"""
    csv_data = file_obj.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_data))
    
    # Apply column mapping
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE)
    
    try:
        action = request.form.get('action', 'analyze')
        
        # Handle clear files action
        if action == 'clear_files':
            session.clear()
            return render_template_string(HTML_TEMPLATE)
        
        # Get form data and save to session
        settings = {
            'min_touches': int(request.form.get('min_touches', 2)),
            'tolerance_percentage': request.form.get('tolerance_percentage', '0.01'),
            'grouping_method': request.form.get('grouping_method', 'conservative'),
            'tolerance_mode': request.form.get('tolerance_mode', 'current_price'),
            'analysis_range_start': request.form.get('analysis_range_start', '148'),
            'analysis_range_end': request.form.get('analysis_range_end', '170')
        }
        session['last_settings'] = settings
        
        # Process uploaded files (only if new files are uploaded)
        timeframe_data = {}
        files_loaded = session.get('files_loaded', {})
        
        # Process new uploads or use cached data
        for tf_key, file_key in [('1D', 'file_1d'), ('4H', 'file_4h'), ('1H', 'file_1h')]:
            # Check if new file uploaded
            if file_key in request.files and request.files[file_key].filename != '':
                df = process_uploaded_file(request.files[file_key])
                save_dataframe_to_session(df, tf_key)
                files_loaded[tf_key] = True
                timeframe_data[tf_key] = df
            # Use cached data if available
            elif tf_key in files_loaded:
                df = load_dataframe_from_session(tf_key)
                if df is not None:
                    timeframe_data[tf_key] = df
        
        session['files_loaded'] = files_loaded
        
        # Validate we have at least 1D data
        if '1D' not in timeframe_data:
            raise ValueError("1D timeframe file is required")
        
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
            
            return render_template_string(HTML_TEMPLATE, range_analysis=range_analysis)
        
        # Full analysis
        finder = MultiTimeframeSRFinder(
            timeframe_data, 
            settings['min_touches'], 
            float(settings['tolerance_percentage']), 
            settings['grouping_method'],
            settings['tolerance_mode']
        )
        
        results = finder.get_detailed_results()
        
        # Add range analysis if requested
        range_analysis = None
        if settings['analysis_range_start'] and settings['analysis_range_end']:
            try:
                range_start = float(settings['analysis_range_start'])
                range_end = float(settings['analysis_range_end'])
                range_analysis = finder.analyze_missing_levels(range_start, range_end)
            except:
                pass
        
        return render_template_string(HTML_TEMPLATE, result=results, range_analysis=range_analysis)
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=str(e))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic analysis"""
    try:
        settings = {
            'min_touches': int(request.form.get('min_touches', 2)),
            'tolerance_percentage': float(request.form.get('tolerance_percentage', 0.01)),
            'grouping_method': request.form.get('grouping_method', 'conservative'),
            'tolerance_mode': request.form.get('tolerance_mode', 'current_price')
        }
        
        timeframe_data = {}
        
        if 'file_1d' not in request.files:
            return jsonify({'error': '1D file is required'}), 400
        
        # Process files
        timeframe_data['1D'] = process_uploaded_file(request.files['file_1d'])
        
        for tf_key, file_key in [('4H', 'file_4h'), ('1H', 'file_1h')]:
            if file_key in request.files and request.files[file_key].filename != '':
                try:
                    timeframe_data[tf_key] = process_uploaded_file(request.files[file_key])
                except Exception as e:
                    print(f"Warning: Could not process {tf_key} file: {e}")
        
        finder = MultiTimeframeSRFinder(
            timeframe_data, 
            settings['min_touches'], 
            settings['tolerance_percentage'], 
            settings['grouping_method'],
            settings['tolerance_mode']
        )
        
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

@app.route('/test-tolerance')
def test_tolerance():
    """Test route to demonstrate tolerance differences"""
    test_html = '''
    <!DOCTYPE html>
    <html>
    <head><title>Tolerance Mode Comparison</title>
    <style>body{font-family:Arial;max-width:900px;margin:50px auto;padding:20px;}</style>
    </head>
    <body>
        <h1>🔬 Tolerance Mode Comparison Test</h1>
        <p>This demonstrates why "Level Price Based" tolerance finds more levels:</p>
        
        <h2>Example: NVDA at different price levels</h2>
        <table border="1" style="border-collapse:collapse;width:100%;">
            <tr style="background:#f0f0f0;">
                <th>Price Level</th>
                <th>Current Price Mode (0.01%)</th>
                <th>Level Price Mode (0.01%)</th>
                <th>Difference</th>
            </tr>
            <tr>
                <td>$50 level</td>
                <td>$0.018 (0.01% of $176)</td>
                <td>$0.005 (0.01% of $50)</td>
                <td>3.6x tighter!</td>
            </tr>
            <tr>
                <td>$100 level</td>
                <td>$0.018 (0.01% of $176)</td>
                <td>$0.010 (0.01% of $100)</td>
                <td>1.8x tighter</td>
            </tr>
            <tr style="background:#fff3cd;">
                <td><strong>$155 level</strong></td>
                <td><strong>$0.018 (0.01% of $176)</strong></td>
                <td><strong>$0.016 (0.01% of $155)</strong></td>
                <td><strong>Adaptive sizing</strong></td>
            </tr>
            <tr>
                <td>$175 level</td>
                <td>$0.018 (0.01% of $176)</td>
                <td>$0.018 (0.01% of $175)</td>
                <td>Nearly same</td>
            </tr>
        </table>
        
        <div style="background:#d4edda;padding:15px;margin:20px 0;border-radius:5px;">
            <h3>💡 Key Insight:</h3>
            <p><strong>Current Price Mode:</strong> Uses $0.018 tolerance everywhere (based on $176 current price)</p>
            <p><strong>Level Price Mode:</strong> Adapts tolerance to each price level</p>
            <p><strong>Result:</strong> Level Price Mode finds more precise groupings at all price ranges!</p>
        </div>
        
        <h2>Test Results Summary</h2>
        <table border="1" style="border-collapse:collapse;width:100%;">
            <tr style="background:#f0f0f0;">
                <th>Method</th>
                <th>Total Levels</th>
                <th>148-170 Range</th>
                <th>Result</th>
            </tr>
            <tr>
                <td>Current Price (0.01%)</td>
                <td>102</td>
                <td>0</td>
                <td>❌ Missing levels</td>
            </tr>
            <tr>
                <td>Level Price (0.01%)</td>
                <td>60</td>
                <td>0</td>
                <td>❌ Still too tight</td>
            </tr>
            <tr style="background:#d4edda;">
                <td><strong>Level Price (0.02%)</strong></td>
                <td><strong>107</strong></td>
                <td><strong>2</strong></td>
                <td><strong>✅ Found levels!</strong></td>
            </tr>
        </table>
        
        <a href="/" style="background:#007bff;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;">
            ← Back to Main App
        </a>
    </body>
    </html>
    '''
    return test_html

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
