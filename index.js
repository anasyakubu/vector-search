const dotenv = require("dotenv").config();
const { mongoose } = require("mongoose");

// database connection
mongoose
  .connect(process.env.MONGODB_URL, {
    dbName: "echelon-ecommerce-db",
    // bufferCommands: false,
    // connectTimeoutMS: 30000,
  })
  .then(() => console.log("Database Connected"))
  .catch((err) => console.log("Database not connected", err));
