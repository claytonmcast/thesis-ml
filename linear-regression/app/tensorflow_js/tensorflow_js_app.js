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

async function fetchDataset(datasetPath, targetColumn, featureColumn) {
    const response = await fetch('/linear-regression/datasets/' + datasetPath);
    const text = await response.text();
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(header => {
        return header.trim().replace(/[\r\n\u00A0]+/g, '');
      });
       
    const data = lines.slice(1).map(line => {
        const values = line.split(',');
        const obj = {};
        headers.forEach((header, index) => {
            obj[header] = parseFloat(values[index]); // Parse as float
        });
        return obj;
    });

    let target = data.map(row => [row[targetColumn]]);
    let features = data.map(row => [row[featureColumn]]);
    return { features, target };
}
  
async function saveData(features, target, predictions, lossHistory, trainingTime, inferenceTime, mse, r2, fileName) {
    var predArray =  predictions.arraySync();
    console.log("predictions", predArray) 
    downloadJson({
        features,
        target,
        predictions: predArray,
        loss_history: lossHistory,
        training_time_ms: trainingTime,
        inference_time_ms: inferenceTime,
        mse,
        r2
    }, fileName.replace('.csv', ''))  
}

function normalizeData(features) {
    const mean = tf.mean(features);
    const std = tf.moments(features).variance.sqrt();
    const normalizedFeatures = tf.div(tf.sub(features, mean), std);
    return { normalizedFeatures: normalizedFeatures.arraySync(), featureScaler: { mean: mean.arraySync(), std: std.arraySync() } };
}

function normalize(train) {
    const mean = tf.mean(train);
    const std = tf.moments(train).variance.sqrt();
    const normalized = tf.div(tf.sub(train, mean), std);
    return normalized.arraySync();
}

function r2Score(labels, predictions) {
    const ssRes = tf.sum(tf.square(tf.sub(labels, predictions)));
    const ssTot = tf.sum(tf.square(tf.sub(labels, tf.mean(labels))));
    return tf.sub(1, tf.div(ssRes, ssTot));
}
 
async function trainModel(features, target) {
    const model = tf.sequential(); 

    model.add(tf.layers.dense({ units: 1, inputShape: [1] })); // Input shape is 1 because we are only training on one feature

    model.compile({ optimizer: tf.train.sgd(0.01), loss: 'meanSquaredError' });
 
    const xs = tf.tensor2d(features);
    const ys = tf.tensor2d(target);

    const lossHistory = [];
    const startTime = performance.now();
    //4096
    await model.fit(xs, ys, { epochs: 200, batchSize: 4096, 
    //await model.fit(xs, ys, { epochs: 200, 
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            lossHistory.push(logs.loss);
           // console.log(`Epoch ${epoch + 1}, Loss: ${logs.loss}`);
          },
        }
    });
    const endTime = performance.now();
    const trainingTime = (endTime - startTime);

    return { model, trainingTime, lossHistory } ;
}
 
async function run(engine, datasetPath) {
    await setupBackend(engine);
    // Fetch the dataset from the API
    const { features, target } = await fetchDataset(datasetPath, "price", "area");

    console.log("features", features);
    console.log("target", target);
    // Normalize the data
    const { normalizedFeatures, featureScaler } = normalizeData(features);
    //const normalizedFeatures = normalize(features);
    const normalizedTarget = target;// normalize(target);
    console.log("normalizedFeatures", normalizedFeatures);
    console.log("normalizedTarget", normalizedTarget);
 
    console.log("normalizedFeatures", normalizedFeatures);
    // Train the model
    const { model, trainingTime, lossHistory } = await trainModel(normalizedFeatures, normalizedTarget);

    // Evaluate the model and get predictions
    const startTime = performance.now(); // Record start time
    const predictions = model.predict(tf.tensor2d(normalizedFeatures));
    await predictions.data(); // Wait for predictions to be available
    const endTime = performance.now(); // Record end time
    const inferenceTime = endTime - startTime;
    
    // Evaluate and print MSE and R2
    const mse = tf.losses.meanSquaredError(normalizedTarget, predictions).arraySync();
    const r2 = r2Score(tf.tensor2d(normalizedTarget), predictions).arraySync();
    console.log(`Mean Squared Error: ${mse}`);
    console.log(`R-squared: ${r2}`);
 
    // Plot the regression line (you'll need a plotting library like Plotly.js or Chart.js)
    await saveData(normalizedFeatures, normalizedTarget, predictions, lossHistory, trainingTime, inferenceTime, mse, r2, "tensorflow_js_" + engine + "_" + datasetPath.replace("house_price/", ""));
  
    return { model }; 
}
 
document.addEventListener('DOMContentLoaded', (event) => {
    Array.from(document.querySelectorAll("button.lr-tf")).forEach(el => {
        el.addEventListener("click", async () => {
            var dsPath = el.getAttribute('dataset');
            var engine = el.getAttribute('engine');
            startProcess(el);
            let result = await run(engine, dsPath);
            stopProcess(el);
        });
    });
});