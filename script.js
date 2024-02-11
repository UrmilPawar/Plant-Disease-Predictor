function predict() {
    var form = document.getElementById('predictionForm');
    var formData = new FormData();

    var fileInput = document.getElementById('image');
    formData.append('image', fileInput.files[0]);
    
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = 'Predicted Disease: ' + data.prediction;
    })
    .catch(error => console.error('Error:', error));
}