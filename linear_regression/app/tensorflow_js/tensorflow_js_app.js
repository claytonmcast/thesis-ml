// Set up TensorFlow.js backend
async function setupBackend(engine) {
    console.log(engine);
    
    // Check if the requested backend is supported
    if (tf.engine().backendNames().includes(engine)) {
        await tf.setBackend(engine);
        await tf.ready();

        // Force backend initialization with a small operation
        const tensor = tf.tensor([0]);
        await tensor.square().data();
    } else {
        console.log(`${engine} not supported, falling back to default backend`);
    }

    console.log(`Using TensorFlow.js backend: ${engine}`);
}

// Load and parse dataset
async function fetchDataset(datasetPath, targetColumn, featureColumn) {
    const response = await fetch('/linear_regression/datasets/' + datasetPath);
    const text = await response.text();
    const lines = text.trim().split('\n');

    const headers = lines[0].split(',').map(header =>
        header.trim().replace(/[\r\n\u00A0]+/g, '')
    );

    const data = lines.slice(1).map(line => {
        const values = line.split(',');
        const obj = {};
        headers.forEach((header, index) => {
            obj[header] = parseFloat(values[index]); // Parse values as floats
        });
        return obj;
    });

    const target = data.map(row => [row[targetColumn]]);
    const features = data.map(row => [row[featureColumn]]);
    
    return { features, target };
}

// Save experiment data and metadata
async function saveData(engine, sample, start, end, features, target, predictions, lossHistory, trainingTime, inferenceTime, mse, r2, fileName) {
    const predArray = predictions.arraySync();
    const experiments_path = "linear_regression/training_result/" + currentResultItem.id;

    await appendExperiment({
        experiment: {
            try: executionTries,
            type: `Linear Regression TensorFlow.js ${engine}`,
            sample,
            title: `Linear Regression TensorFlow.js ${engine} ${sample}`,
            start,
            end,
            platform: `tensorflow_js_${engine}`,
            result_item_id: currentResultItem.id,
            location: experiments_path,
            try_path: `${experiments_path}/${executionTries}`,
            experiment_path: `${experiments_path}/${executionTries}/tensorflow_js_${engine}`,
            result_path: `${experiments_path}/${executionTries}/${fileName.replace('.csv', '')}.json`
        },
        results: {
            features,
            target,
            predictions: predArray,
            loss_history: lossHistory,
            training_time_ms: trainingTime,
            inference_time_ms: inferenceTime,
            mse,
            r2
        }
    });
}

// Normalize feature values using z-score normalization
function normalizeData(features) {
    const mean = tf.mean(features);
    const std = tf.moments(features).variance.sqrt();
    const normalizedFeatures = tf.div(tf.sub(features, mean), std);

    return {
        normalizedFeatures: normalizedFeatures.arraySync(),
        featureScaler: {
            mean: mean.arraySync(),
            std: std.arraySync()
        }
    };
}

// Normalize a single array
function normalize(train) {
    const mean = tf.mean(train);
    const std = tf.moments(train).variance.sqrt();
    const normalized = tf.div(tf.sub(train, mean), std);
    return normalized.arraySync();
}

// Calculate R-squared score
function r2Score(labels, predictions) {
    const ssRes = tf.sum(tf.square(tf.sub(labels, predictions)));
    const ssTot = tf.sum(tf.square(tf.sub(labels, tf.mean(labels))));
    return tf.sub(1, tf.div(ssRes, ssTot));
}

// Train a linear regression model
async function trainModel(features, target) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    model.compile({
        optimizer: tf.train.sgd(0.01),
        loss: 'meanSquaredError'
    });

    const xs = tf.tensor2d(features);
    const ys = tf.tensor2d(target);

    const lossHistory = [];
    const startTime = performance.now();

    await model.fit(xs, ys, {
        epochs: 200,
        batchSize: 4096,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                lossHistory.push(logs.loss);
            }
        }
    });

    const endTime = performance.now();
    const trainingTime = endTime - startTime;
    xs.dispose();
    ys.dispose();

    return { model, trainingTime, lossHistory };
}

// Orchestrates the full linear regression flow
async function runLinearRegression(engine, datasetPath, sample) {
    const startProcessTs = new Date();

    await setupBackend(engine);
   // tf.engine().startScope();

    // Load and prepare data
    const { features, target } = await fetchDataset(datasetPath, "price", "area");
    const { normalizedFeatures } = normalizeData(features);
    const normalizedTarget = target;

    // Train the model
    const { model, trainingTime, lossHistory } = await trainModel(normalizedFeatures, normalizedTarget);

    // Run predictions
    const startTime = performance.now();
    const predictions = model.predict(tf.tensor2d(normalizedFeatures));
    await predictions.data(); // Ensure predictions are resolved
    const endTime = performance.now();
    const inferenceTime = endTime - startTime;

    // Evaluate performance
    const mse = tf.losses.meanSquaredError(normalizedTarget, predictions).arraySync();
    const r2 = r2Score(tf.tensor2d(normalizedTarget), predictions).arraySync();
    console.log(`Mean Squared Error: ${mse}`);
    console.log(`R-squared: ${r2}`);


    // Save results
    const endProcessTs = new Date();
    await saveData(engine, sample, startProcessTs, endProcessTs, normalizedFeatures, normalizedTarget, predictions, lossHistory, trainingTime, inferenceTime, mse, r2,
        `tensorflow_js_${engine}/tensorflow_js_${engine}_${datasetPath.replace("house_price/", "")}`
    );


    model.dispose();
    //tf.engine().endScope();
    tf.disposeVariables();
    
    return { };
}

// Button click handler for running regression
const handleLinearRegression = async (el, position) => {
    const dsPath = el.getAttribute('dataset');
    const engine = el.getAttribute('engine');
    const sample = el.getAttribute('sample');

    if (runAllProcessing !== true) {
        await getNewResultItem();
    }

    startProcess(el);
    await startProcessing(el, async () => await runLinearRegression(engine, dsPath, sample), position);
    if (runAllProcessing !== true) {
        await plotLinearRegression();
    }
    stopProcess(el);
};

// Attach click listeners to buttons once DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    Array.from(document.querySelectorAll("button.lr-tf")).forEach(el => {
        el.addEventListener("click", async () => {
            await handleLinearRegression(el);
        });
    });
});

export default handleLinearRegression;
