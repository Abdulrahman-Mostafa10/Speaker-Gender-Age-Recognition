const fs = require("fs").promises;
const path = require("path");

async function deleteCsvFiles(dir) {
  try {
    const files = await fs.readdir(dir, { withFileTypes: true });

    for (const file of files) {
      const fullPath = path.join(dir, file.name);

      if (file.isDirectory()) {
        // Recursively process subdirectories
        await deleteCsvFiles(fullPath);
      } else if (
        file.isFile() &&
        path.extname(file.name).toLowerCase() === ".csv"
      ) {
        await fs.unlink(fullPath);
        console.log(`Deleted: ${fullPath}`);
      }
    }
  } catch (err) {
    console.error(`Error processing ${dir}:`, err);
  }
}

deleteCsvFiles("../data_processed/Female_Fifties");
