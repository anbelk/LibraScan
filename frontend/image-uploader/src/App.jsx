import React, { useState, useRef } from 'react';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictedClass, setPredictedClass] = useState(null);
  const [showManualLabeling, setShowManualLabeling] = useState(false);
  const [selectedManualClass, setSelectedManualClass] = useState(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleImageChange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      setSelectedImage(imageUrl);
      resetState();
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.match('image.*')) {
      const imageUrl = URL.createObjectURL(file);
      setSelectedImage(imageUrl);
      resetState();
    }
  };

  const resetState = () => {
    setPredictedClass(null);
    setShowManualLabeling(false);
    setSelectedManualClass(null);
    setFeedbackSubmitted(false);
  };

  const handleSubmit = async () => {
    if (!selectedImage) return;

    try {
      // Получаем файл из input
      const file = fileInputRef.current.files[0];
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setPredictedClass(data.class);
      setShowManualLabeling(true);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handleManualLabel = (classLabel) => {
    setSelectedManualClass(classLabel);
  };

  const submitFeedback = () => {
    setFeedbackSubmitted(true);
    setShowManualLabeling(false);
    console.log('Final class:', selectedManualClass);
  };

  const classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4'];

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 p-4">
      <div className="w-full max-w-md bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-bold text-center mb-6">Image Classifier</h1>

        {/* Область загрузки изображения */}
        <div
          className={`flex flex-col items-center mb-6 cursor-pointer ${isDragging ? 'bg-blue-50' : ''}`}
          onClick={triggerFileInput}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="hidden"
            ref={fileInputRef}
          />

          {selectedImage ? (
            <div className="relative w-full">
              <img
                src={selectedImage}
                alt="Preview"
                className="w-full h-48 object-contain border-2 border-gray-200 rounded-lg"
                style={{ maxHeight: '200px' }}
              />
            </div>
          ) : (
            <div className={`w-full flex flex-col items-center px-4 py-6 rounded-lg border-2 border-dashed transition-colors 
              ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-blue-300 hover:bg-blue-50'}`}>
              <svg className="w-12 h-12 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
              </svg>
              <span className="mt-2 text-base font-medium text-gray-700">
                {isDragging ? 'Отпустите для загрузки' : 'Нажмите или перетащите изображение'}
              </span>
              <span className="text-sm text-gray-500">Поддерживаемые форматы: PNG, JPG, JPEG</span>
            </div>
          )}
        </div>

        {!predictedClass && (
          <button
            onClick={handleSubmit}
            disabled={!selectedImage}
            className={`w-full py-3 px-4 rounded-md text-white font-medium transition-colors
              ${selectedImage ? 'bg-blue-500 hover:bg-blue-600' : 'bg-gray-400 cursor-not-allowed'}`}
          >
            Анализировать изображение
          </button>
        )}

        {predictedClass && !feedbackSubmitted && (
          <div className="space-y-4">
            <div className="mt-6 p-4 bg-blue-50 rounded-md">
              <h2 className="text-lg font-semibold text-center">
                Результат анализа: <span className="text-blue-600">{predictedClass}</span>
              </h2>
            </div>

            {showManualLabeling && (
              <div className="mt-6">
                <h3 className="text-lg font-medium text-center mb-4">Разметить вручную:</h3>
                <div className="grid grid-cols-2 gap-2">
                  {classes.map((classLabel) => (
                    <button
                      key={classLabel}
                      onClick={() => handleManualLabel(classLabel)}
                      className={`py-2 px-4 rounded-md transition-colors
                        ${selectedManualClass === classLabel ? 
                          'bg-green-500 text-white' : 
                          'bg-gray-200 hover:bg-gray-300'}`}
                    >
                      {classLabel}
                    </button>
                  ))}
                </div>
                {selectedManualClass && (
                  <button
                    onClick={submitFeedback}
                    className="w-full mt-4 py-2 px-4 bg-green-500 text-white rounded-md hover:bg-green-600 transition-colors"
                  >
                    Подтвердить выбор
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        {feedbackSubmitted && (
          <div className="mt-6 p-4 bg-green-50 text-green-700 text-center rounded-md">
            Спасибо за размеченные данные!
          </div>
        )}
      </div>
    </div>
  );
}

export default App;