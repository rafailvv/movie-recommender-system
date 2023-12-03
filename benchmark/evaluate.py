import time

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from benchmark.data.dataset import MovieDataset
import pickle


class RecommenderNet(nn.Module):
    def __init__(
        self,
        num_zip_codes,
        num_release_years,
        num_occupations,
        num_genres,
        embedding_size,
    ):
        super(RecommenderNet, self).__init__()
        # Embeddings
        self.zip_code_embedding = nn.Embedding(num_zip_codes, embedding_size)
        self.release_year_embedding = nn.Embedding(num_release_years, embedding_size)
        self.occupation_embedding = nn.Embedding(num_occupations, embedding_size)

        # Linear layers for age and gender
        self.age_lin = nn.Linear(1, embedding_size)
        self.gender_lin = nn.Linear(2, embedding_size)

        # Fully connected layers
        self.fc1 = nn.Linear(embedding_size * 5 + num_genres, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

    def forward(
        self, zip_codes, release_years, ages, occupations, genders, genre_features
    ):
        zip_code_embedding = self.zip_code_embedding(zip_codes)
        release_year_embedding = self.release_year_embedding(release_years)
        occupation_embedding = self.occupation_embedding(occupations)
        age_embedding = self.age_lin(ages.unsqueeze(1))
        gender_embedding = self.gender_lin(genders)

        x = torch.cat(
            [
                zip_code_embedding,
                release_year_embedding,
                occupation_embedding,
                age_embedding,
                gender_embedding,
                genre_features,
            ],
            dim=1,
        )
        x = nn.ReLU()(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = nn.ReLU()(self.bn2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()


df = pd.read_csv("data/preprocessed.csv")
all_genres = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

with open("../models/occupation_encoder.pkl", "rb") as file:
    occupation_encoder = pickle.load(file)
with open("../models/zip_code_encoder.pkl", "rb") as file:
    zip_code_encoder = pickle.load(file)
with open("../models/release_year_encoder.pkl", "rb") as file:
    release_year_encoder = pickle.load(file)

# occupation_encoder = LabelEncoder()
# zip_code_encoder = LabelEncoder()
# release_year_encoder = LabelEncoder()
# df['zip_code'] = zip_code_encoder.fit_transform(df['zip_code'])
# df['release_year'] = release_year_encoder.fit_transform(df['release_year'])
# df['occupation'] = occupation_encoder.fit_transform(df['occupation'])

embedding_size = 50
batch_size = 32
num_zip_codes = df["zip_code"].nunique()
num_release_years = df["release_year"].nunique()
num_movies = df["film_id"].nunique()
num_occupations = df["occupation"].nunique()
num_genres = len(all_genres)
model = RecommenderNet(
    num_zip_codes, num_release_years, num_occupations, num_genres, embedding_size
)
model.load_state_dict(torch.load("../models/model.pth"))


def recommend_movies(model, age, gender, occupation, zip_code, num_recommendations=5):
    # Convert inputs using encoders and normalization
    encoded_occupation = occupation_encoder.transform([occupation])[0]
    encoded_zip_code = zip_code_encoder.transform([zip_code])[0]
    normalized_age = (age - df["age"].mean()) / df["age"].std()

    # Prepare gender input
    gender_input = np.array([[1, 0] if gender == "F" else [0, 1]])

    # Prepare inputs for all movies
    movie_ids = np.arange(num_movies)
    zip_codes = np.full_like(movie_ids, encoded_zip_code)
    ages = np.full_like(movie_ids, normalized_age, dtype=np.float32)
    occupations = np.full_like(movie_ids, encoded_occupation)
    genders = np.tile(gender_input, (num_movies, 1))

    # Convert to tensors
    zip_codes_tensor = torch.tensor(zip_codes)
    ages_tensor = torch.tensor(ages, dtype=torch.float32)
    occupations_tensor = torch.tensor(occupations)
    genders_tensor = torch.tensor(genders, dtype=torch.float32)

    # Generate predictions for all movies
    model.eval()
    predictions = np.zeros(num_movies)
    with torch.no_grad():
        for i in range(0, num_movies, batch_size):
            if i + batch_size > num_movies:
                i = num_movies - batch_size
            genre_features = torch.tensor(
                df.loc[:, list(all_genres)].values, dtype=torch.float32
            )
            release_years = torch.tensor(df["release_year"].tolist())
            batch_predictions = model(
                zip_codes_tensor[i : i + batch_size],
                release_years[i : i + batch_size],
                ages_tensor[i : i + batch_size],
                occupations_tensor[i : i + batch_size],
                genders_tensor[i : i + batch_size],
                genre_features[i : i + batch_size],
            )
            predictions[i : i + batch_size] = batch_predictions.numpy()

    # Sort by predicted rating
    sorted_indices = np.argsort(predictions)[::-1]
    top_movie_ids = sorted_indices[:num_recommendations]
    top_movie_ratings = predictions[top_movie_ids]

    # Convert movie IDs back to movie names and pair with their ratings
    movies_data = []
    for movie_id, rating in zip(top_movie_ids, top_movie_ratings * 5):
        movie_title = df[df["film_id"] == movie_id]["title"].iloc[0]
        movies_data.append({"id": movie_id, "title": movie_title, "rating": rating})

    # Convert to DataFrame
    recommended_movies_df = pd.DataFrame(movies_data)
    recommended_movies_df.index = range(1, len(recommended_movies_df) + 1)
    return recommended_movies_df


def evaluate_model(model, test_loader):
    model.eval()
    mse = 0
    with torch.no_grad():
        for (
            zip_codes,
            release_years,
            ages,
            occupations,
            genders,
            genre_features,
        ), ratings in test_loader:
            outputs = (
                model(
                    zip_codes, release_years, ages, occupations, genders, genre_features
                )
                * 5
            )
            mse += mse_loss(outputs, ratings)
    mse = mse / len(test_loader)
    return np.sqrt(mse)


# Example user details
while True:
    menu = int(
        input(
            """
Menu:
1. Enter user data (age, gender, occupation, zip code) manually
2. Enter id of existed user.
3. Calculate RMSE for all dataset
4. Exit

Enter 1, 2, 3 or 4: """
        )
    )

    if menu == 2:
        user_df = pd.read_csv("../data/interim/user.csv")
        user_id = int(
            input(f"Enter user_id (from 1 to {len(user_df['user_id'].tolist())}):")
        )
        if 1 <= user_id <= len(user_df["user_id"].tolist()):
            user_details = user_df[user_df["user_id"] == user_id]
            age = user_details["age"].iloc[0]
            gender = user_details["gender"].iloc[0]
            occupation = user_details["occupation"].iloc[0]
            zip_code = user_details["zip_code"].iloc[0]

            print(f"Age: {age}")
            print(f"Gender: {gender}")
            print(f"Occupation: {occupation}")
            print(f"Zip Code: {zip_code}")
        else:
            print(f"User ID {user_id} not found in the dataset.")
            continue
    elif menu == 1:
        age = int(input("Age: "))
        gender = input("Gender M (male) or F (female): ")
        occupation = input("Occupation: ")
        zip_code = input("Zip code: ")
    elif menu == 3:
        test_loader = DataLoader(MovieDataset(df), batch_size=batch_size)
        rmse = evaluate_model(model, test_loader)
        print(f"RMSE on test set: {rmse}")
        time.sleep(1)
        continue
    else:
        exit()

    try:
        # Number of recommendations
        num_recommendations = int(input("Enter number of recommendations: "))

        # Get recommendations
        top_movies_df = recommend_movies(
            model, age, gender, occupation, zip_code, num_recommendations
        )
        print(top_movies_df)
    except:
        print(
            "Input error, most likely not contained in the dataset. Please try again."
        )
