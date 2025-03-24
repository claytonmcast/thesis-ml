async function setupBackend(engine) {
    console.log(engine);
    if (tf.engine().backendNames().includes(engine)) {
        await tf.setBackend(engine);
        await tf.ready();
        const tensor = tf.tensor([0]);
        await tensor.square().data();
    } else {
        console.log(`${engine} not supported, falling back to default backend`);
    }
    console.log(`Using TensorFlow.js backend: ${engine}`);
}

async function loadMNIST(trainPercentage = 1.0) {
    const path = 'neural-network/datasets/';
    const trainImagesResponse = await fetch(path + 'mnist_train_images.json');
    const trainImagesData = await trainImagesResponse.json();
    const trainLabelsResponse = await fetch(path + 'mnist_train_labels.json');
    const trainLabelsData = await trainLabelsResponse.json();
    const testImagesResponse = await fetch(path + 'mnist_test_images.json');
    const testImagesData = await testImagesResponse.json();
    const testLabelsResponse = await fetch(path + 'mnist_test_labels.json');
    const testLabelsData = await testLabelsResponse.json();

    // Convert to TensorFlow.js tensors
    let trainImages = tf.tensor2d(trainImagesData);
    let trainLabels = tf.tensor2d(trainLabelsData);
    let testImages = tf.tensor2d(testImagesData);
    let testLabels = tf.tensor2d(testLabelsData);

    // Calculate sample sizes
    const numTrainSamples = Math.floor(trainImages.shape[0] * trainPercentage);
    const numTestSamples = Math.floor(testImages.shape[0] * trainPercentage);

    // Slice the data
    trainImages = trainImages.slice([0, 0], [numTrainSamples, 784]);
    trainLabels = trainLabels.slice([0, 0], [numTrainSamples, 10]);
    testImages = testImages.slice([0, 0], [numTestSamples, 784]);
    testLabels = testLabels.slice([0, 0], [numTestSamples, 10]);

    return { trainImages, trainLabels, testImages, testLabels };
}

async function predictAndMeasure(model, inputTensor) {
    // Start time
    const startTime = performance.now();

    // Make prediction
    const predictions = model.predict(inputTensor);

    // End time
    const endTime = performance.now();

    // Calculate inference time
    const inferenceTime = endTime - startTime;

    // Get predicted class
    const predictedClassTensor = predictions.argMax(1);
    const predictedClass = (await predictedClassTensor.array())[0];

    // Dispose of tensors to free memory
    predictions.dispose();
    predictedClassTensor.dispose();

    return { predictedClass, inferenceTime };
}


// Preprocess Data
async function trainModel(engine, trainPercentage) {
    const { trainImages, trainLabels, testImages, testLabels } = await loadMNIST(trainPercentage);

    // Build and train your model here (using trainImages, trainLabels, etc.)
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [784] }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    model.compile({
        optimizer: tf.train.sgd(0.01),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const lossValues = [];
    const accuracyValues = [];
    const valLossValues = [];
    const valAccuracyValues = [];

    const startTime = performance.now();
    await model.fit(trainImages, trainLabels, {
        epochs: 10,
        validationData: [testImages, testLabels],
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                lossValues.push(logs.loss);
                accuracyValues.push(logs.acc);
                valLossValues.push(logs.val_loss);
                valAccuracyValues.push(logs.val_acc);
                // console.log(
                // `Epoch ${epoch + 1}: loss = ${logs.loss}, acc = ${logs.acc}, val_loss = ${
                //     logs.val_loss
                // }, val_acc = ${logs.val_acc}`
                // );
            },
        },
    });
    const endTime = performance.now();
    const trainingTime = (endTime - startTime);
    console.log('Training time:', trainingTime, 'milliseconds');

    const evalResult = await model.evaluate(testImages, testLabels);

    const lossTensor = evalResult[0];
    const accuracyTensor = evalResult[1];

    const loss = await lossTensor.array();
    const accuracy = await accuracyTensor.array();

    console.log('Loss:', loss);
    console.log('Accuracy:', accuracy);

    const sampleImage = testImages.slice([0], [1]);

    const { predictedClass, inferenceTime } = await predictAndMeasure(model, sampleImage);

    console.log('Predicted class:', predictedClass);
    console.log('Inference time:', inferenceTime);

    downloadJson({
        loss_values: lossValues,
        accuracy_values: accuracyValues,
        val_loss_values: valLossValues,
        val_accuracy_values:valAccuracyValues,
        training_time_ms: trainingTime,
        inference_time_ms: inferenceTime,
        loss,
        accuracy
    }, 'nn_mnist_tensorflow_js_' + engine +'_sample_' + (trainPercentage * 100) + "%")  
}


document.addEventListener('DOMContentLoaded', (event) => {
    Array.from(document.querySelectorAll("button.mnist-tf")).forEach(el => {
        el.addEventListener("click", async () => {
            var trainPercentage = el.getAttribute('dataset');
            var engine = el.getAttribute('engine');

            startProcess(el);
            await setupBackend(engine);
            await trainModel(engine, parseFloat(trainPercentage));
            stopProcess(el);
        });
    });
});