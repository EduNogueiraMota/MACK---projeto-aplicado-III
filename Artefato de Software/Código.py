#Importando as Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


#Carregar o dataset book
books = pd.read_csv("C:/Users/biazi/OneDrive - Instituto Presbiteriano Mackenzie/Mackenzie/Aulas/4° Semestre/Projeto Aplicado III/Datasets/books.csv", on_bad_lines='skip', engine="python")

# Ver as primeiras linhas
print(books.head())

# Verificar a estrutura do dataset
print(books.info())

# Descrever as estatísticas do dataset
print(books.describe())

#Verificando os valores ausentes
print(books.isnull().sum())

#Renomeando coluna num_pages
books = books.rename(columns={'  num_pages': 'num_pages'})

# Histograma das avaliações médias
plt.hist(books['average_rating'], bins=50)
plt.xlabel('Avaliação Média')
plt.ylabel('Frequência')
plt.title('Distribuição das Avaliações Médias')
plt.show()

# Histograma do número de páginas
plt.hist(books['num_pages'].dropna(), bins=50)
plt.xlabel('Número de Páginas')
plt.ylabel('Frequência')
plt.title('Distribuição do Número de Páginas')
plt.show()

# Livros mais avaliados
most_rated_books = books.nlargest(10, 'ratings_count')
print(most_rated_books[['title', 'authors', 'ratings_count']])

# Scatter plot
plt.scatter(books['average_rating'], books['num_pages'])
plt.xlabel('Avaliação Média')
plt.ylabel('Número de Páginas')
plt.title('Relação entre Avaliação Média e Número de Páginas')
plt.show()

#Contagem de livros por Idioma
language_counts = books['language_code'].value_counts()
print(language_counts)

# Carregar o dataset ratings
ratings = pd.read_csv("C:/Users/biazi/OneDrive - Instituto Presbiteriano Mackenzie/Mackenzie/Aulas/4° Semestre/Projeto Aplicado III/Datasets/ratings.csv", on_bad_lines='skip', engine="python")

# Ver as primeiras linhas
print(ratings.head())

# Verificar a estrutura do dataset
print(ratings.info())

# Descrever as estatísticas do dataset
print(ratings.describe())

# Verificar valores ausentes
print(ratings.isnull().sum())

# Histograma das classificações
plt.hist(ratings['rating'], bins=5, edgecolor='black')
plt.xlabel('Classificação')
plt.ylabel('Frequência')
plt.title('Distribuição das Classificações')
plt.show()

# Classificações médias por livro
average_ratings_per_book = ratings.groupby('book_id')['rating'].mean()
print(average_ratings_per_book.describe())

# Número de classificações por usuário
ratings_per_user = ratings.groupby('user_id').size()
print(ratings_per_user.describe())

# Criar uma matriz de usuários e livros
user_book_matrix = ratings.pivot_table(index='user_id',
                                    columns='book_id',
                                    values='rating',
                                    aggfunc='first') # Use pivot_table with aggfunc to handle duplicates

# Preencher valores NaN com 0
user_book_matrix.fillna(0, inplace=True)


# Calcular a média das avaliações e o número de avaliações para cada livro
book_ratings = ratings.groupby('book_id')['rating'].agg(['mean', 'count']).reset_index()

# Filtrar os livros com pelo menos 50 avaliações para garantir uma base de avaliação mais confiável
book_ratings = book_ratings[book_ratings['count'] >= 50]

# Classificar os livros pela média das avaliações, do maior para o menor
book_ratings = book_ratings.sort_values(by='mean', ascending=False)

# Selecionar os top N livros
num_recommendations = 5
top_books = book_ratings.head(num_recommendations)

# Obter os detalhes dos livros recomendados
# Changed 'book_id' to 'bookID' to match the column name in the 'books' DataFrame
recommended_books = books[books['bookID'].isin(top_books['book_id'])]

print(recommended_books[['title', 'authors', 'average_rating']])

# Dividir os dados em conjuntos de treino e teste
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Calcular a média das avaliações no conjunto de treino
train_mean_ratings = train_data.groupby('book_id')['rating'].agg(['mean', 'count']).reset_index()
train_mean_ratings = train_mean_ratings[train_mean_ratings['count'] >= 50]
train_mean_ratings = train_mean_ratings.sort_values(by='mean', ascending=False)

# Selecionar os top N livros do conjunto de treino
top_books_train = train_mean_ratings.head(num_recommendations)

# Predizer as avaliações no conjunto de teste usando as médias do treino
test_data['predicted_rating'] = test_data['book_id'].apply(lambda x: train_mean_ratings.loc[train_mean_ratings['book_id'] == x, 'mean'].values[0] if x in train_mean_ratings['book_id'].values else 0)

# Avaliar o desempenho do modelo usando Mean Squared Error
mse = mean_squared_error(test_data['rating'], test_data['predicted_rating'])
print(f'Mean Squared Error: {mse:.4f}')

# Carregar o modelo SVD da biblioteca Surprise
model = SVD()

# Definir o leitor para o dataset
reader = Reader(rating_scale=(1, 5))

# Criar o dataset Surprise a partir do DataFrame de avaliações
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)

# Treinar o modelo com validação cruzada
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Treinar o modelo com todos os dados de treinamento
trainset = data.build_full_trainset()
model.fit(trainset)

# Prever a avaliação de um usuário para um livro específico
user_id = 1  # ID do usuário
book_id = 10 # ID do livro
prediction = model.predict(user_id, book_id)
print(f'Previsão da avaliação do usuário {user_id} para o livro {book_id}: {prediction.est}')
