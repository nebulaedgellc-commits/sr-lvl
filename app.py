from flask import Flask, request, jsonify, session
import pandas as pd
import numpy as np
import io
import os

app = Flask(__name__)
app.secret_key = 'sr-levels-2024'

class SimpleSRFinder:
    def __init__(self, df, min_touches=2, tolerance_percentage=0.01, tolerance_mode="current_price"):
        self.df = df
        self.min_touches = min_touches
        self.tolerance_percentage = tolerance_percentage / 100.0
        self.tolerance_mode = tolerance_mode
        self.prepare_data()
        
    def prepare_data(self):
        # Clean column names
        self.df.columns = self.df.columns.str.lower().str.strip()
        
        # Map columns
        column_map = {
            'time': 'date', 'datetime': 'date',
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
        }
        self.df.rename(columns=column_map, inplace=True)
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")
        
        # Clean data
        self.df.dropna(inplace=True)
        self.current_price = self.df['close'].iloc[-1]
        self.base_tolerance = self.current_price * self.tolerance_percentage
        
        print(f"Current price: ${self.current_price:.2f}")
        print(f"Tolerance mode: {self.tolerance_mode}")
        print(f"Base tolerance: ${self.base_tolerance:.3f}")
    
    def get_tolerance(self, price):
        if self.tolerance_mode == "current_price":
            return self.base_tolerance
        else:
            return price * self.tolerance_percentage
    
    def group_prices(self, prices, level_type):
        if len(prices) == 0:
            return []
        
        sorted_prices = sorted(prices)
        groups = []
        current_group = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            group_avg = sum(current_group) / len(current_group)
            tolerance = self.get_tolerance(group_avg)
            min_distance = min(abs(price - p) for p in current_group)
            
            if min_distance <= tolerance:
                current_group.append(price)
            else:
                if len(current_group) >= self.min_touches:
                    level_price = sum(current_group) / len(current_group)
                    groups.append({
                        'level': level_price,
                        'type': level_type,
                        'touches': len(current_group),
                        'tolerance': self.get_tolerance(level_price)
                    })
                current_group = [price]
        
        # Add final group
        if len(current_group) >= self.min_touches:
            level_price = sum(current_group) / len(current_group)
            groups.append({
                'level': level_price,
                'type': level_type,
                'touches': len(current_group),
                'tolerance': self.get_tolerance(level_price)
            })
        
        return groups
    
    def find_levels(self):
        highs = self.df['high'].tolist()
        lows = self.df['low'].tolist()
        
        resistance_levels = self.group_prices(highs, 'Resistance')
        support_levels = self.group_prices(lows, 'Support')
        
        all_levels = resistance_levels + support_levels
        all_levels.sort(key=lambda x: x['touches'], reverse=True)
        
        return all_levels
    
    def analyze_range(self, start_price, end_price):
        levels = self.find_levels()
        range_levels = [l for l in levels if start_price <= l['level'] <= end_price]
        
        # Find all prices in range
        all_prices = []
        for _, row in self.df.iterrows():
            if start_price <= row['high'] <= end_price:
                all_prices.append(row['high'])
            if start_price <= row['low'] <= end_price:
                all_prices.append(row['low'])
        
        unique_prices = sorted(list(set(all_prices)))
        
        analysis = {
            'range': f"${start_price}-${end_price}",
            'levels_found': len(range_levels),
            'price_occurrences': len(all_prices),
            'unique_prices': len(unique_prices),
            'current_tolerance': self.base_tolerance
        }
        
        if len(unique_prices) > 1:
            min_gap = min(unique_prices[i+1] - unique_prices[i] for i in range(len(unique_prices)-1))
            mid_price = (start_price + end_price) / 2
            suggested_tolerance = (min_gap / mid_price) * 100
            analysis['min_gap'] = min_gap
            analysis['suggested_tolerance_pct'] = suggested_tolerance
        
        return analysis, range_levels

# Simple HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Simple S&R Level Finder</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        .section { border: 1px solid #ccc; padding: 15px; margin: 15px 0; border-radius: 5px; }
        .radio-group { margin: 10px 0; }
        .radio-option { margin: 5px 0; padding: 10px; border: 1px solid #eee; border-radius: 3px; }
        .recommended { background: #f0fff0; border-color: #28a745; }
        input[type="file"], input[type="number"] { width: 100%; padding: 8px; margin: 5px 0; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        .result { background: #f8f9fa; padding: 15px; margin: 15px 0; border-left: 4px solid #007bff; }
        .error { background: #f8d7da; color: #721c24; border-left-color: #dc3545; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f2f2f2; }
        .levels-output { font-family: monospace; background: #e9ecef; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🎯 Simple S&R Level Finder</h1>
    <p><strong>Features:</strong> Level-based tolerance, Radio buttons, File persistence</p>
    
    <form method="post" enctype="multipart/form-data">
        <div class="section">
            <h3>📁 Upload CSV File</h3>
            <input type="file" name="csv_file" accept=".csv" required>
            <p><small>Upload CSV with columns: time,open,high,low,close,volume</small></p>
        </div>
        
        <div class="section">
            <h3>⚙️ Settings</h3>
            
            <div class="radio-group">
                <label><strong>Tolerance Mode:</strong></label>
                <div class="radio-option">
                    <input type="radio" name="tolerance_mode" value="current_price" checked>
                    <label>Current Price Based (Original)</label>
                </div>
                <div class="radio-option recommended">
                    <input type="radio" name="tolerance_mode" value="level_price">
                    <label><strong>Level Price Based (RECOMMENDED)</strong> - Fixes missing levels</label>
                </div>
            </div>
            
            <div class="radio-group">
                <label><strong>Tolerance Percentage:</strong></label>
                <div class="radio-option">
                    <input type="radio" name="tolerance_percentage" value="0.005">
                    <label>0.005% - Ultra Tight</label>
                </div>
                <div class="radio-option">
                    <input type="radio" name="tolerance_percentage" value="0.01" checked>
                    <label>0.01% - Tight (Default)</label>
                </div>
                <div class="radio-option recommended">
                    <input type="radio" name="tolerance_percentage" value="0.02">
                    <label><strong>0.02% - Balanced (NVDA Recommended)</strong></label>
                </div>
                <div class="radio-option">
                    <input type="radio" name="tolerance_percentage" value="0.03">
                    <label>0.03% - Loose</label>
                </div>
            </div>
            
            <div class="radio-group">
                <label><strong>Minimum Touches:</strong></label>
                <div class="radio-option">
                    <input type="radio" name="min_touches" value="1">
                    <label>1 Touch - Maximum Sensitivity</label>
                </div>
                <div class="radio-option recommended">
                    <input type="radio" name="min_touches" value="2" checked>
                    <label><strong>2 Touches - Good Default</strong></label>
                </div>
                <div class="radio-option">
                    <input type="radio" name="min_touches" value="3">
                    <label>3 Touches - Conservative</label>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>🔍 Range Analysis (Optional)</h3>
            <label>From Price:</label>
            <input type="number" name="range_start" step="0.01" placeholder="148.00">
            <label>To Price:</label>
            <input type="number" name="range_end" step="0.01" placeholder="170.00">
        </div>
        
        <button type="submit">🎯 Find S&R Levels</button>
    </form>
    
    {% if result %}
    <div class="result">
        <h3>🎯 Results</h3>
        <p><strong>Total levels found:</strong> {{ result.total_count }}</p>
        <p><strong>Current price:</strong> ${{ "%.2f"|format(result.current_price) }}</p>
        <p><strong>Tolerance mode:</strong> {{ result.tolerance_mode.replace('_', ' ').title() }}</p>
        <p><strong>Base tolerance:</strong> ${{ "%.3f"|format(result.base_tolerance) }}</p>
        
        <h4>📋 Levels for TradingView:</h4>
        <div class="levels-output">{{ result.levels_csv }}</div>
        
        <h4>📊 Level Details:</h4>
        <table>
            <tr><th>Level</th><th>Type</th><th>Touches</th><th>Tolerance</th></tr>
            {% for level in result.levels[:10] %}
            <tr>
                <td>${{ "%.2f"|format(level.level) }}</td>
                <td>{{ level.type }}</td>
                <td>{{ level.touches }}</td>
                <td>${{ "%.3f"|format(level.tolerance) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
    
    {% if range_analysis %}
    <div class="result">
        <h3>🔍 Range Analysis: {{ range_analysis.range }}</h3>
        <p><strong>Levels found in range:</strong> {{ range_analysis.levels_found }}</p>
        <p><strong>Price occurrences in range:</strong> {{ range_analysis.price_occurrences }}</p>
        <p><strong>Unique prices in range:</strong> {{ range_analysis.unique_prices }}</p>
        
        {% if range_analysis.get('suggested_tolerance_pct') %}
        <div style="background: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <strong>💡 Suggested Solution:</strong><br>
            Use <strong>{{ "%.3f"|format(range_analysis.suggested_tolerance_pct) }}%</strong> tolerance to capture levels in this range<br>
            (Minimum gap between prices: ${{ "%.3f"|format(range_analysis.min_gap) }})
        </div>
        {% endif %}
        
        {% if range_levels %}
        <h4>Levels found in {{ range_analysis.range }}:</h4>
        <table>
            <tr><th>Level</th><th>Type</th><th>Touches</th></tr>
            {% for level in range_levels %}
            <tr>
                <td>${{ "%.2f"|format(level.level) }}</td>
                <td>{{ level.type }}</td>
                <td>{{ level.touches }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
    {% endif %}
    
    {% if error %}
    <div class="result error">
        <h3>❌ Error</h3>
        <p>{{ error }}</p>
    </div>
    {% endif %}
</body>
</html>
'''

def process_csv_file(file_obj):
    """Process uploaded CSV file"""
    try:
        csv_data = file_obj.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return HTML_TEMPLATE
    
    try:
        # Get form data
        tolerance_mode = request.form.get('tolerance_mode', 'current_price')
        tolerance_percentage = float(request.form.get('tolerance_percentage', 0.01))
        min_touches = int(request.form.get('min_touches', 2))
        range_start = request.form.get('range_start')
        range_end = request.form.get('range_end')
        
        # Process CSV file
        if 'csv_file' not in request.files or request.files['csv_file'].filename == '':
            return HTML_TEMPLATE.replace('{% if error %}', '{% if True %}').replace('{{ error }}', 'Please upload a CSV file')
        
        csv_file = request.files['csv_file']
        df = process_csv_file(csv_file)
        
        # Analyze levels
        finder = SimpleSRFinder(df, min_touches, tolerance_percentage, tolerance_mode)
        levels = finder.find_levels()
        
        # Prepare results
        levels_csv = ",".join([f"{level['level']:.2f}" for level in levels])
        
        result = {
            'total_count': len(levels),
            'current_price': finder.current_price,
            'tolerance_mode': tolerance_mode,
            'base_tolerance': finder.base_tolerance,
            'levels_csv': levels_csv,
            'levels': levels
        }
        
        # Range analysis if requested
        range_analysis = None
        range_levels = None
        if range_start and range_end:
            try:
                start = float(range_start)
                end = float(range_end)
                range_analysis, range_levels = finder.analyze_range(start, end)
            except:
                pass
        
        # Render template with results
        template = HTML_TEMPLATE
        template = template.replace('{% if result %}', '{% if True %}')
        template = template.replace('{{ result.total_count }}', str(result['total_count']))
        template = template.replace('{{ "%.2f"|format(result.current_price) }}', f"{result['current_price']:.2f}")
        template = template.replace('{{ result.tolerance_mode.replace(\'_\', \' \').title() }}', result['tolerance_mode'].replace('_', ' ').title())
        template = template.replace('{{ "%.3f"|format(result.base_tolerance) }}', f"{result['base_tolerance']:.3f}")
        template = template.replace('{{ result.levels_csv }}', result['levels_csv'])
        
        # Add level details
        level_rows = ""
        for level in levels[:10]:
            level_rows += f"<tr><td>${level['level']:.2f}</td><td>{level['type']}</td><td>{level['touches']}</td><td>${level['tolerance']:.3f}</td></tr>"
        template = template.replace('{% for level in result.levels[:10] %}', '').replace('{% endfor %}', level_rows)
        
        # Range analysis
        if range_analysis:
            template = template.replace('{% if range_analysis %}', '{% if True %}')
            template = template.replace('{{ range_analysis.range }}', range_analysis['range'])
            template = template.replace('{{ range_analysis.levels_found }}', str(range_analysis['levels_found']))
            template = template.replace('{{ range_analysis.price_occurrences }}', str(range_analysis['price_occurrences']))
            template = template.replace('{{ range_analysis.unique_prices }}', str(range_analysis['unique_prices']))
            
            if 'suggested_tolerance_pct' in range_analysis:
                template = template.replace('{% if range_analysis.get(\'suggested_tolerance_pct\') %}', '{% if True %}')
                template = template.replace('{{ "%.3f"|format(range_analysis.suggested_tolerance_pct) }}', f"{range_analysis['suggested_tolerance_pct']:.3f}")
                template = template.replace('{{ "%.3f"|format(range_analysis.min_gap) }}', f"{range_analysis['min_gap']:.3f}")
            
            if range_levels:
                template = template.replace('{% if range_levels %}', '{% if True %}')
                range_level_rows = ""
                for level in range_levels:
                    range_level_rows += f"<tr><td>${level['level']:.2f}</td><td>{level['type']}</td><td>{level['touches']}</td></tr>"
                template = template.replace('{% for level in range_levels %}', '').replace('{% endfor %}', range_level_rows, 1)
        
        # Clean up remaining template syntax
        template = template.replace('{% if False %}', '<!--').replace('{% endif %}', '-->')
        template = template.replace('{% if result %}', '<!--').replace('{% if range_analysis %}', '<!--')
        template = template.replace('{% if range_levels %}', '<!--').replace('{% if error %}', '<!--')
        
        return template
        
    except Exception as e:
        error_template = HTML_TEMPLATE.replace('{% if error %}', '{% if True %}').replace('{{ error }}', str(e))
        error_template = error_template.replace('{% if result %}', '<!--').replace('{% if range_analysis %}', '<!--').replace('{% endif %}', '-->')
        return error_template

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/test')
def test():
    return '<h1>Simple S&R App Working!</h1><p>Go to <a href="/">/</a> to use the app.</p>'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
