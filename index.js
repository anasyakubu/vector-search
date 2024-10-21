const dotenv = require("dotenv").config();
const { mongoose, Schema } = require("mongoose");
const axios = require("axios"); // For calling Hugging Face API
const hfTOKEN = process.env.HF_TOKEN;

const embeddingUrl =
  "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2";

// Connect to the sample_mflix database
mongoose
  .connect(process.env.MONGODB_URL, {
    dbName: "sample_mflix", // Use the sample_mflix database
  })
  .then(() => console.log("Database Connected"))
  .catch((err) => console.log("Database not connected", err));

// Define movie schema
const movieSchema = new Schema({
  title: String,
  description: String,
  genre: [String],
  year: Number,
  embeddings: {
    type: [Number], // This will store the embeddings
    default: [],
  },
});

// Ensure the collection name is explicitly set to 'movies'
const Movie = mongoose.model("Movie", movieSchema, "movies");

// Function to create embeddings for a movie
const createEmbedding = async (text) => {
  try {
    const response = await axios.post(
      embeddingUrl,
      { inputs: text },
      { headers: { Authorization: `Bearer ${hfTOKEN}` } }
    );
    return response.data[0]; // The embedding vector
  } catch (error) {
    console.error("Error generating embedding:", error);
  }
};

// Function to store movie and embedding in MongoDB
const storeMovieWithEmbedding = async (title, description, genre, year) => {
  const textToEmbed = `${title} ${description}`;
  const embedding = await createEmbedding(textToEmbed);

  const newMovie = new Movie({
    title,
    description,
    genre,
    year,
    embeddings: embedding, // Store the generated embeddings
  });

  await newMovie.save();
  console.log(`Movie "${title}" with embeddings stored in DB.`);
};

// Create an index on the embeddings field for vector search
const createVectorIndex = async () => {
  await Movie.collection.createIndex(
    { embeddings: "vector" }, // Use 'vector' as index type
    { v: 2, dimensions: 384 } // Set dimensions based on the model
  );
  console.log("Vector index created on embeddings field.");
};

// Query the database using vector search
const searchMovies = async (queryText) => {
  const queryEmbedding = await createEmbedding(queryText); // Create embedding for the query text

  if (!queryEmbedding) {
    console.error("Embedding not generated, cannot perform search.");
    return;
  }

  const results = await Movie.aggregate([
    {
      $search: {
        index: "default", // Specify the vector index
        knnBeta: {
          vector: queryEmbedding, // The generated embedding vector
          path: "embeddings", // The path to the embeddings field in your documents
          k: 5, // Return top 5 nearest neighbors
        },
      },
    },
  ]);

  console.log("Search results:", results);
};

// Example: Call this function to store a new movie with embeddings
// storeMovieWithEmbedding(
//   "Interstellar",
//   "A sci-fi movie about space travel and time",
//   ["Sci-Fi", "Adventure"],
//   2014
// );

// Example: Call this function to create vector index
// createVectorIndex();

// Example: Searching for a movie similar to the query text
searchMovies("A sci-fi movie with a futuristic space adventure");
