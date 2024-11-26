import { useState } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [similarImages, setSimilarImages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [textQuery, setTextQuery] = useState("");

  const handleImageUpload = (event) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedImage(event.target.files[0]);
    }
  };

  const handleTextQueryChange = (event) => {
    setTextQuery(event.target.value);
  };

  const uploadImage = async () => {
    if (!selectedImage) {
      alert("Please select an image to upload.");
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      setIsLoading(true);
      const response = await axios.post("http://localhost:5000/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      alert(response.data.message);
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Failed to upload image.");
    } finally {
      setIsLoading(false);
    }
  };

  const searchSimilarImagesByImage = async () => {
    if (!selectedImage) {
      alert("Please select an image to search.");
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      setIsLoading(true);
      const response = await axios.post("http://localhost:5000/query_image", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setSimilarImages(response.data.similar_images);
    } catch (error) {
      console.error("Error querying similar images:", error);
      alert("Failed to search similar images.");
    } finally {
      setIsLoading(false);
    }
  };

  const searchSimilarImagesByText = async () => {
    if (!textQuery.trim()) {
      alert("Please enter a text query.");
      return;
    }

    try {
      setIsLoading(true);
      const response = await axios.post("http://localhost:5000/query_text", { query: textQuery }, {
        headers: {
          "Content-Type": "application/json",
        },
      });
      setSimilarImages(response.data.similar_images);
    } catch (error) {
      console.error("Error querying similar images by text:", error);
      alert("Failed to search similar images.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Content based Image Retrieval (CBIR)</h1>
      </header>

      <main className="app-main">
        <div className="upload-section">
          <input type="file" onChange={handleImageUpload} className="file-input" />
          {selectedImage && (
            <div className="selected-image">
              <h3>Selected Image:</h3>
              <img
                src={URL.createObjectURL(selectedImage)}
                alt="Selected"
                className="preview-image"
              />
            </div>
          )}
          <div className="buttons">
            <button onClick={uploadImage} className="upload-button" disabled={isLoading}>
              {isLoading ? 'Uploading...' : 'Upload Image'}
            </button>
            <button onClick={searchSimilarImagesByImage} className="search-button" disabled={isLoading}>
              {isLoading ? 'Searching...' : 'Search Similar Images'}
            </button>
          </div>
        </div>

        <div className="text-query-section">
          <h3>Search by Text:</h3>
          <input
            type="text"
            value={textQuery}
            onChange={handleTextQueryChange}
            placeholder="Enter your query here..."
            className="text-input"
          />
          <button onClick={searchSimilarImagesByText} className="search-text-button" disabled={isLoading}>
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {similarImages.length > 0 && (
          <div className="results-section">
            <h2>Similar Images:</h2>
            <div className="images-grid">
              {similarImages.map((img, index) => (
                <div key={index} className="image-card">
                  <img
                    src={`http://localhost:5000/uploads/animal2/${img.image_path}`}
                    alt={img.image_name}
                    className="result-image"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = '/placeholder.jpg';
                    }}
                  />
                  <p className="similarity-text">Similarity: {img.similarity.toFixed(2)}%</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;