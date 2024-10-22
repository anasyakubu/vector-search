require("dotenv").config();
const express = require("express");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const fs = require("fs");
const path = require("path");
const mongoose = require("mongoose");
const { HfInference } = require("@huggingface/inference");

// Initialize Hugging Face API
const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

// Connect to MongoDB
mongoose
  .connect(process.env.MONGODB_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("MongoDB connected"))
  .catch((err) => console.log(err));

// Create a schema for embeddings
const embeddingSchema = new mongoose.Schema({
  documentName: String,
  embedding: [Number], // Store the embedding as an array of numbers
});

const Embedding = mongoose.model("Embedding", embeddingSchema);

// Initialize Express app
const app = express();
app.use(express.json());

// Configure Multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  },
});

const upload = multer({ storage });

// Endpoint to upload PDF and process it
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    const filePath = path.join(__dirname, "uploads", req.file.filename);

    // Extract text from the PDF
    const dataBuffer = fs.readFileSync(filePath);
    const pdfData = await pdfParse(dataBuffer);
    const extractedText = pdfData.text;

    // Remove the file after processing
    fs.unlinkSync(filePath);

    // Generate embeddings for the extracted text
    const embeddings = await hf.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: extractedText,
    });

    // Save the embeddings to MongoDB
    const newEmbedding = new Embedding({
      documentName: req.file.filename,
      embedding: embeddings[0], // Save the first embedding from the array
    });
    await newEmbedding.save();

    res.status(201).json({ message: "Embedding saved successfully" });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Error processing the PDF" });
  }
});

// Endpoint to query the embeddings
app.post("/query", async (req, res) => {
  try {
    const { queryText } = req.body;

    // Generate embedding for the query
    const queryEmbedding = await hf.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: queryText,
    });

    // Perform a simple nearest neighbor search (for demonstration purposes)
    const allEmbeddings = await Embedding.find();
    let bestMatch = null;
    let highestSimilarity = -Infinity;

    allEmbeddings.forEach((doc) => {
      const similarity = cosineSimilarity(queryEmbedding[0], doc.embedding);
      if (similarity > highestSimilarity) {
        highestSimilarity = similarity;
        bestMatch = doc;
      }
    });

    res.json({
      message: `Closest match for your query: ${bestMatch.documentName}`,
      similarity: highestSimilarity,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Error processing query" });
  }
});

// Utility function to calculate cosine similarity
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
