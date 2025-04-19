// Setup the backend for TensorFlow.js
async function setupBackend(engine) {
    console.log(engine); // Log the selected engine
    await tf.setBackend('cpu');
    await tf.ready();
    await sleep(1000);
    if (tf.engine().backendNames().includes(engine)) {
        // If the engine is supported, set it as the backend
        await tf.setBackend(engine);
        await tf.ready();
        const tensor = tf.tensor([0]);
        await tensor.square().data(); // Test the backend by running a simple operation
    } else {
        // Log if the engine is not supported and fallback to the default backend
        console.log(`${engine} not supported, falling back to default backend`);
    }
    console.log(`Using TensorFlow.js backend: ${engine}`);
}

// Load MNIST dataset (images and labels) and preprocess it
async function loadMNIST(trainPercentage = 1.0) {
    const path = 'neural_network/datasets/';
    
    // Fetch the dataset files (train and test images and labels)
    const trainImagesResponse = await fetch(path + 'mnist_train_images.json');
    const trainImagesData = await trainImagesResponse.json();
    const trainLabelsResponse = await fetch(path + 'mnist_train_labels.json');
    const trainLabelsData = await trainLabelsResponse.json();
    const testImagesResponse = await fetch(path + 'mnist_test_images.json');
    const testImagesData = await testImagesResponse.json();
    const testLabelsResponse = await fetch(path + 'mnist_test_labels.json');
    const testLabelsData = await testLabelsResponse.json();

    // Convert JSON data to TensorFlow.js tensors
    let trainImages = tf.tensor2d(trainImagesData);
    let trainLabels = tf.tensor2d(trainLabelsData);
    let testImages = tf.tensor2d(testImagesData);
    let testLabels = tf.tensor2d(testLabelsData);

    // Calculate the number of samples based on the training percentage
    const numTrainSamples = Math.floor(trainImages.shape[0] * trainPercentage);
    const numTestSamples = Math.floor(testImages.shape[0] * trainPercentage);

    // Slice the data according to the selected percentage
    trainImages = trainImages.slice([0, 0], [numTrainSamples, 784]);
    trainLabels = trainLabels.slice([0, 0], [numTrainSamples, 10]);
    testImages = testImages.slice([0, 0], [numTestSamples, 784]);
    testLabels = testLabels.slice([0, 0], [numTestSamples, 10]);

    return { trainImages, trainLabels, testImages, testLabels };
}

// Predict and measure inference time
async function predictAndMeasure(model, inputTensor) {
    // Start the timer for inference time
    const startTime = performance.now();

    // Make the prediction using the model
    const predictions = model.predict(inputTensor);

    // End the timer for inference time
    const endTime = performance.now();

    // Calculate inference time
    const inferenceTime = endTime - startTime;

    // Get the predicted class (the class with the highest probability)
    const predictedClassTensor = predictions.argMax(1);
    const predictedClass = (await predictedClassTensor.array())[0];

    // Dispose of tensors to free memory
    predictions.dispose();
    predictedClassTensor.dispose();

    return { predictedClass, inferenceTime };
}

// Train the model with the given engine and data
async function trainModel(engine, trainPercentage, sample) {
    const startProcessMs = new Date();
    //tf.engine().startScope();
    
    // Setup the TensorFlow.js backend
    await setupBackend(engine);
    // Load the MNIST dataset
    const { trainImages, trainLabels, testImages, testLabels } = await loadMNIST(trainPercentage);

    // Build the model architecture (simple feedforward neural network)
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [784] }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    // Compile the model
    model.compile({
        optimizer: tf.train.sgd(0.01),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Arrays to store loss and accuracy values during training
    const lossValues = [];
    const accuracyValues = [];
    const valLossValues = [];
    const valAccuracyValues = [];

    // Start training and record the time taken
    const startTime = performance.now();
    await model.fit(trainImages, trainLabels, {
        epochs: 10,
        validationData: [testImages, testLabels],
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                // Log the loss and accuracy values at the end of each epoch
                lossValues.push(logs.loss);
                accuracyValues.push(logs.acc);
                valLossValues.push(logs.val_loss);
                valAccuracyValues.push(logs.val_acc);
            },
        },
    });
    const endTime = performance.now();
    const trainingTime = (endTime - startTime); // Calculate total training time
    
    // Evaluate the model on the test dataset
    const evalResult = await model.evaluate(testImages, testLabels);

    // Extract the loss and accuracy from evaluation
    const lossTensor = evalResult[0];
    const accuracyTensor = evalResult[1];
    const loss = await lossTensor.array();
    const accuracy = await accuracyTensor.array();

     
    // Use a sample image to test the model's inference time
    const sampleImage = testImages.slice([0], [1]);
    const { predictedClass, inferenceTime } = await predictAndMeasure(model, sampleImage);
        
    const endProcessMs = new Date();

    // Path for storing experiment results
    const experimentsPath = "neural_network/training_result/" + currentResultItem.id;

    // Append the experiment results to the experiment history
    await appendExperiment({
        experiment: {
            try: executionTries,
            type: "Neural Network Rust TensorFlow.js " + engine,
            sample,
            title: "Neural Network TensorFlow.js " + engine + " " + sample,
            start: startProcessMs,
            end: endProcessMs,
            platform: "tensorflow_js_" + engine,
            result_item_id: currentResultItem.id,
            location: experimentsPath,
            try_path: experimentsPath + "/" + executionTries,
            experiment_path: experimentsPath + "/" + executionTries + "/tensorflow_js_" + engine,
            result_path: experimentsPath + "/" + executionTries + "/tensorflow_js_" + engine + "/nn_mnist_tensorflow_js_" + engine + "_sample_" + (trainPercentage * 100) + "%.json"
        },
        results: {
            loss_values: lossValues,
            accuracy_values: accuracyValues,
            val_loss_values: valLossValues,
            val_accuracy_values: valAccuracyValues,
            training_time_ms: trainingTime,
            inference_time_ms: inferenceTime,
            loss,
            accuracy
        }
    });

    model.dispose();
    trainImages.dispose();
    trainLabels.dispose();
    testImages.dispose();
    testLabels.dispose();
    lossTensor.dispose();
    accuracyTensor.dispose();
    sampleImage.dispose();
   // tf.engine().endScope();
    tf.disposeVariables();
}

// Handle the neural network execution when a button is clicked
const handleNeuralNetwork = async (el, position) => {
    var trainPercentage = el.getAttribute('dataset');
    var engine = el.getAttribute('engine');
    var sample = el.getAttribute('sample');

    // Start the processing indicator
    startProcess(el);

    // If not processing all, fetch a new result item
    if (runAllProcessing !== true) {
        await getNewResultItem();
    }

    // Start training the model
    await startProcessing(el, async () => await trainModel(engine, parseFloat(trainPercentage), sample), position);

    // If not processing all, plot the results
    if (runAllProcessing !== true) {
        await plotNeuralNetwork();
    }

    // Stop the processing indicator
    stopProcess(el);
};

// Add event listeners to trigger the neural network training when a button is clicked
document.addEventListener('DOMContentLoaded', (event) => {
    Array.from(document.querySelectorAll("button.mnist-tf")).forEach(el => {
        el.addEventListener("click", async () => {
            await handleNeuralNetwork(el);
        });
    });
});

// Export the handler for use in other modules
export default handleNeuralNetwork;
