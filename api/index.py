from flask import Flask, render_template, request
import pandas as pd
from collections import defaultdict

app = Flask(__name__)

# Load dataset
df = pd.read_csv('recipes.csv', usecols=['name', 'ingredients', 'steps', 'description', 'tags'])

df['tags'] = df['tags'].fillna('')
df['ingredients'] = df['ingredients'].fillna('')

df['searchable_text'] = df['tags'] + ' ' + df['ingredients']

# Simple keyword indexing
keyword_index = defaultdict(list)

for idx, row in df.iterrows():
    text = row['searchable_text'].lower()  # Convert to lowercase for case-insensitive matching
    for keyword in text.split():
        keyword_index[keyword].append(row.to_dict())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_recipe', methods=['POST'])
def generate_recipe():
    search_input = request.form['search-input'].strip().lower()  # Remove leading/trailing spaces and convert to lowercase
    keywords = search_input.split()

    matching_recipes = []

    for keyword in keywords:
        if keyword in keyword_index:
            matching_recipes.extend(keyword_index[keyword])

    if matching_recipes:
        matching_recipes = sorted(matching_recipes, key=lambda x: sum(1 for kw in keywords if kw in x['searchable_text']), reverse=True)

    recipes = matching_recipes

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return render_template('recipes_partial.html', recipes=recipes)

    return render_template('index.html', recipes=recipes)

if __name__ == '__main__':
    app.run(debug=True)
