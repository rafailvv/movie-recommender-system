import torch
from torch.utils.data import Dataset

all_genres = ["unknown", "Action", "Adventure", "Animation", "Children's",
              "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
              "Sci-Fi", "Thriller", "War", "Western"]
class MovieDataset(Dataset):
    def __init__(self, dataframe):
        self.ages = torch.tensor(dataframe['age'].values, dtype=torch.float32)
        self.occupations = torch.tensor(dataframe['occupation'].values)
        self.genders = torch.tensor(dataframe[['gender_F', 'gender_M']].values, dtype=torch.float32)
        self.genre_features = torch.tensor(dataframe[list(all_genres)].values, dtype=torch.float32)
        self.zip_codes = torch.tensor(dataframe['zip_code'].values)
        self.release_years = torch.tensor(dataframe['release_year'].values)
        self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (self.zip_codes[idx], self.release_years[idx],
                self.ages[idx], self.occupations[idx],
                self.genders[idx], self.genre_features[idx]), self.ratings[idx]

