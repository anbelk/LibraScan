import React, { useState, useRef } from 'react';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictedClass, setPredictedClass] = useState(null);
  const [showManualLabeling, setShowManualLabeling] = useState(false);
  const [selectedManualClass, setSelectedManualClass] = useState(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.match('image.*')) {
      const imageUrl = URL.createObjectURL(file);
      setSelectedImage(imageUrl);
      resetState();
    }
    // ⚠️ НЕ сбрасываем e.target.value — это вызывает повторное открытие окна выбора файла
  };

  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const resetFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
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
      console.error('Ошибка при отправке:', error);
    }
  };

  const handleManualLabel = (classLabel) => {
    setSelectedManualClass(classLabel);
  };

  const submitFeedback = () => {
    setFeedbackSubmitted(true);
    // Можно оставить выделение, либо сбросить:
    // setSelectedManualClass(null);
  };

  const startNewClassification = () => {
    setSelectedImage(null);
    resetState();
    resetFileInput(); // 🟢 Сбрасываем file input только тут
  };

  const classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4'];

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 p-4">
      <div className="w-full max-w-md bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-bold text-center mb-6">Image Classifier</h1>

        {feedbackSubmitted ? (
          <div className="space-y-4">
            <div className="p-4 bg-green-50 text-green-700 text-center rounded-md">
              Спасибо за оценку!
            </div>
            <button
              onClick={startNewClassification}
              className="w-full py-2 px-4 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
            >
              Загрузить новое изображение
            </button>
          </div>
        ) : (
          <>
            {/* Зона загрузки изображения */}
            <div
              className="flex flex-col items-center mb-6 cursor-pointer"
              onClick={triggerFileInput}
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
                <div className="w-96 flex flex-col items-center px-4 py-6 rounded-lg border-2 border-dashed border-blue-300 hover:bg-blue-50 transition-colors text-center">
                  <span className="mt-2 text-base font-medium text-gray-700 whitespace-nowrap">
                    Нажмите для загрузки изображения
                  </span>
                </div>
              )}
            </div>

            {/* Кнопка анализа */}
            {!predictedClass && selectedImage && (
              <button
                onClick={handleSubmit}
                className="w-full py-3 px-4 bg-blue-500 hover:bg-blue-600 text-white font-medium rounded-md transition-colors"
              >
                Анализировать изображение
              </button>
            )}

            {/* Результат и ручная разметка */}
            {predictedClass && (
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
                          className={`py-2 px-4 rounded-md font-medium border-2 transition-all duration-150
                            ${
                              selectedManualClass === classLabel && !feedbackSubmitted
                                ? 'bg-green-700 text-white border-green-800 shadow-lg scale-105'
                                : 'bg-gray-200 hover:bg-gray-300 text-gray-800 border-transparent'
                            }`}
                        >
                          {classLabel}
                        </button>
                      ))}
                    </div>

                    {selectedManualClass && !feedbackSubmitted && (
                      <button
                        onClick={submitFeedback}
                        className="w-full mt-4 py-2 px-4 bg-green-700 hover:bg-green-800 text-white rounded-md transition-colors"
                      >
                        Подтвердить выбор
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App;
