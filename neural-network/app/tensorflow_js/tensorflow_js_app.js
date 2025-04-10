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
async function trainModel(engine, trainPercentage, sample) {
    const startProcessMs = new Date();
    await setupBackend(engine);
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
            },
        },
    });
    const endTime = performance.now();
    const trainingTime = (endTime - startTime);

    const evalResult = await model.evaluate(testImages, testLabels);

    const lossTensor = evalResult[0];
    const accuracyTensor = evalResult[1];

    const loss = await lossTensor.array();
    const accuracy = await accuracyTensor.array();

    const sampleImage = testImages.slice([0], [1]);

    const { predictedClass, inferenceTime } = await predictAndMeasure(model, sampleImage);

    const endProcessMs = new Date();

    const experiments_path = "neural-network/training_result/" + currentResultItem.id;
    
    appendExperiment({
        experiment: {
            try: executionTries,
            type: "Neural Network Rust TensorFlow.js " + engine ,
            sample,
            title: "Neural Network TensorFlow.js " + engine + " " + sample,
            start: startProcessMs,
            end: endProcessMs,
            result_item_id: currentResultItem.id,
            location: experiments_path,
            experiment_path: experiments_path + "/" + executionTries,
            result_path:  experiments_path + "/" + executionTries + "/" + 'nn_mnist_tensorflow_js_' + engine +'_sample_' + (trainPercentage * 100) + "%.json"
        },
        results: {
            loss_values: lossValues,
            accuracy_values: accuracyValues,
            val_loss_values: valLossValues,
            val_accuracy_values:valAccuracyValues,
            training_time_ms: trainingTime,
            inference_time_ms: inferenceTime,
            loss,
            accuracy
        }
    })   
}


const handleNeuralNetwork = async (el) => {
    var trainPercentage = el.getAttribute('dataset');
    var engine = el.getAttribute('engine');
    var sample = el.getAttribute('sample');

    if(runAllProcessing != true){
        getNewResultItem();
    }
    startProcess(el);
    await startProcessing(el, async ()=> await trainModel(engine, parseFloat(trainPercentage), sample))
    stopProcess(el);
};

document.addEventListener('DOMContentLoaded', (event) => {
    Array.from(document.querySelectorAll("button.mnist-tf")).forEach(el => {
        el.addEventListener("click", async () => {
            await handleNeuralNetwork(el)
        });
    });
});

export default handleNeuralNetwork;