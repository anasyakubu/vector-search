require("dotenv").config();
const express = require("express");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const fs = require("fs");
const path = require("path");
const mongoose = require("mongoose");
const { HfInference } = require("@huggingface/inference");
const { OpenAI } = require("openai");

// Check if necessary environment variables are loaded
if (
  !process.env.HUGGINGFACE_API_KEY ||
  !process.env.MONGODB_URL ||
  !process.env.OPENAI_API_KEY
) {
  console.error(
    "Please set the HUGGINGFACE_API_KEY, MONGODB_URL, and OPENAI_API_KEY in your .env file"
  );
  process.exit(1);
}

// Initialize Hugging Face API
const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

// Initialize OpenAI API
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Connect to MongoDB
mongoose
  .connect(process.env.MONGODB_URL, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("MongoDB connected"))
  .catch((err) => console.log("MongoDB connection error:", err));

// Create a schema for embeddings
const embeddingSchema = new mongoose.Schema({
  documentName: String,
  embedding: [Number], // Store the embedding as an array of numbers
  content: String, // Store the extracted text content
});

const Embedding = mongoose.model("Embedding", embeddingSchema);

// Initialize Express app
const app = express();
app.use(express.json());

// Ensure 'uploads' directory exists
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// Configure Multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir); // Ensure the correct path
  },
  filename: (req, file, cb) => {
    console.log(`Uploading file: ${file.originalname}`);
    cb(null, file.originalname); // Use original file name
  },
});

const upload = multer({ storage });

// Endpoint to upload PDF and process it
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    // Check if a file is uploaded
    if (!req.file) {
      console.log("No file uploaded");
      return res.status(400).json({ message: "No file uploaded" });
    }

    console.log("File uploaded successfully:", req.file);

    const filePath = path.join(uploadDir, req.file.filename);

    // Extract text from the PDF
    const dataBuffer = fs.readFileSync(filePath);
    const pdfData = await pdfParse(dataBuffer);
    const extractedText = pdfData.text;

    // Remove the file after processing
    fs.unlinkSync(filePath);

    // Generate embeddings for the extracted text using Hugging Face
    const embeddings = await hf.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: extractedText,
    });

    // Save the embeddings and extracted text to MongoDB
    const newEmbedding = new Embedding({
      documentName: req.file.filename,
      embedding: embeddings[0], // Save the first embedding from the array
      content: extractedText, // Save the extracted text content
    });
    await newEmbedding.save();

    res.status(201).json({ message: "Embedding saved successfully" });
  } catch (error) {
    console.error("Error:", error);
    // Handle Multer-specific errors
    if (error instanceof multer.MulterError) {
      return res
        .status(400)
        .json({ message: `Multer error: ${error.message}` });
    }
    res.status(500).json({ message: "Error processing the PDF" });
  }
});

// Endpoint to query the embeddings and OpenAI
app.post("/query", async (req, res) => {
  try {
    const { queryText } = req.body;

    // Check if the query text is provided
    if (!queryText) {
      return res.status(400).json({ message: "Query text is required" });
    }

    // Generate embedding for the query
    const queryEmbedding = await hf.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: queryText,
    });

    // Perform a simple nearest neighbor search
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

    // If a match is found, send the content to OpenAI
    if (bestMatch) {
      const openAiResponse = await openai.chat.completions.create({
        model: "gpt-3.5-turbo", // Specify your model here
        messages: [
          {
            role: "user",
            content: `Based on the document: ${bestMatch.content}, ${queryText}`,
          },
        ],
      });

      res.json({
        message: `Closest match for your query: ${bestMatch.documentName}`,
        response: openAiResponse.choices[0].message.content,
        similarity: highestSimilarity,
      });
    } else {
      res.json({ message: "No match found", similarity: highestSimilarity });
    }
  } catch (error) {
    console.error("Error:", error);
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
const PORT = process.env.PORT || 8000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
