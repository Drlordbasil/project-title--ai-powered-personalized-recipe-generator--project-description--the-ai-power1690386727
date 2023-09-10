import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class RecipeGenerator:
    def __init__(self, recipe_csv, substitution_csv):
        self.recipes = pd.read_csv(recipe_csv)
        self.substitutions = pd.read_csv(substitution_csv)

    def load_data(self, recipe_csv, substitution_csv):
        self.recipes = pd.read_csv(recipe_csv)
        self.substitutions = pd.read_csv(substitution_csv)

    def preprocess_input(self, user_input):
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(user_input.lower())
        stop_words = nltk.corpus.stopwords.words('english')
        processed_tokens = [
            token for token in tokens if token not in stop_words]
        return ' '.join(processed_tokens)

    def sentiment_score(self, description):
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(description)['compound']

    def compute_tfidf_matrix(self, descriptions):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)
        return tfidf_matrix, tfidf_vectorizer

    def compute_cosine_similarity(self, tfidf_matrix, tfidf_vectorizer, processed_input):
        cosine_sim = linear_kernel(
            tfidf_matrix, tfidf_vectorizer.transform([processed_input])).flatten()
        return cosine_sim

    def recommend_recipes(self, user_input, preferences):
        processed_input = self.preprocess_input(user_input)

        self.recipes['sentiment_score'] = self.recipes['description'].apply(
            self.sentiment_score)

        tfidf_matrix, tfidf_vectorizer = self.compute_tfidf_matrix(
            self.recipes['description'])

        cosine_sim = self.compute_cosine_similarity(
            tfidf_matrix, tfidf_vectorizer, processed_input)

        self.recipes['score'] = 0.6 * \
            self.recipes['sentiment_score'] + 0.4 * cosine_sim
        recommended_recipes = self.recipes.sort_values(
            by='score', ascending=False).head(5)

        for preference in preferences:
            recommended_recipes = recommended_recipes[recommended_recipes[preference] == 1]

        return recommended_recipes[['title', 'description', 'ingredients']]

    def generate_recipe_instructions(self, recipe):
        instructions = recipe['instructions'].split('\n')
        instructions = [step for step in instructions if step.strip() != '']
        return instructions

    def suggest_substitutions(self, ingredient, dietary_restrictions):
        suggested_substitutions = self.substitutions[
            (self.substitutions['ingredient'] == ingredient.lower()) & (
                self.substitutions['dietary_restriction'].isin(dietary_restrictions))
        ]['substitute']

        return suggested_substitutions


if __name__ == '__main__':
    recipe_generator = RecipeGenerator('recipes.csv', 'substitutions.csv')

    user_input = input('Enter your preferences (comma separated): ')
    preferences = ['vegetarian', 'gluten-free']

    recommended_recipes = recipe_generator.recommend_recipes(
        user_input, preferences)

    for _, recipe in recommended_recipes.iterrows():
        print(recipe['title'])
        print(recipe['description'])
        print('\nIngredients:')
        print(recipe['ingredients'])
        print('\nInstructions:')
        instructions = recipe_generator.generate_recipe_instructions(recipe)
        for step in instructions:
            print(step)
        print('\n-----\n')

    ingredient = input('Enter an ingredient: ')
    dietary_restrictions = ['gluten-free']

    suggested_substitutions = recipe_generator.suggest_substitutions(
        ingredient, dietary_restrictions)

    print(f'Suggested substitutions for {ingredient}:')
    for substitution in suggested_substitutions:
        print(substitution)
