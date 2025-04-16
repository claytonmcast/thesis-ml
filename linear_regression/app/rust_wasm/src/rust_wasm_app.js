import init, { fetch_and_train } from "../pkg/rust_wasm.js";

/**
 * Fetches and parses a CSV dataset, extracting the specified feature and target columns.
 * @param {string} datasetPath - Relative path to the dataset file.
 * @param {string} targetColumn - The column name to use as the target (label).
 * @param {string} featureColumn - The column name to use as the feature.
 * @returns {Promise<{features: number[][], target: number[][]}>}
 */
async function fetchDataset(datasetPath, targetColumn, featureColumn) {
    const response = await fetch('/linear_regression/datasets/' + datasetPath);
    const text = await response.text();
    const lines = text.trim().split('\n');
    
    const headers = lines[0].split(',').map(header =>
        header.trim().replace(/[\r\n\u00A0]+/g, '')
    );
    
    const data = lines.slice(1).map(line => {
        const values = line.split(',');
        const row = {};
        headers.forEach((header, index) => {
            row[header] = parseFloat(values[index]);
        });
        return row;
    });

    // Extract feature and target columns into 2D arrays
    const features = data.map(row => [row[featureColumn]]);
    const target = data.map(row => [row[targetColumn]]);
    
    return { features, target };
}

/**
 * Executes the training process using the Rust WASM linear regression engine.
 * @param {string} engine - The engine identifier (not used directly here, but passed in).
 * @param {string} datasetPath - The path to the CSV dataset.
 * @param {string} sample - A sample label used for tracking experiments.
 */
async function run(engine, datasetPath, sample) {
    const startProcessTs = new Date();
    
    await init(); // Initialize the Rust WASM module

    // Load dataset and flatten 2D arrays to 1D arrays as required by Rust
    const { features, target } = await fetchDataset(datasetPath, "price", "area");
    const features1D = features.flat();
    const target1D = target.flat();

    // Call Rust function for training and prediction
    const resultData = await fetch_and_train(features1D, target1D);

    const endProcessTs = new Date();

    // Construct experiment metadata
    const basePath = "linear_regression/training_result/" + currentResultItem.id;
    const experimentName = "Linear Regression Rust WASM CPU";
    const fileName = "rust_wasm_cpu_" + datasetPath.replace("house_price/", "").replace('.csv', '') + '.json';

    appendExperiment({
        experiment: {
            try: executionTries,
            type: experimentName,
            sample,
            title: `${experimentName} ${sample}`,
            start: startProcessTs,
            end: endProcessTs,
            platform: 'rust_wasm_cpu',
            result_item_id: currentResultItem.id,
            location: basePath,
            try_path: `${basePath}/${executionTries}`,
            experiment_path: `${basePath}/${executionTries}/rust_wasm_cpu`,
            result_path: `${basePath}/${executionTries}/rust_wasm_cpu/${fileName}`
        },
        results: resultData
    });
}

/**
 * Event handler for the "Run" button that starts the WASM-based linear regression.
 * @param {HTMLElement} el - The button element that triggered the event.
 */
const handleLinearRegressionRustWasm = async (el) => {
    const datasetPath = el.getAttribute('dataset');
    const engine = el.getAttribute('engine');
    const sample = el.getAttribute('sample');

    if (!runAllProcessing) {
        await getNewResultItem();
    }

    startProcess(el);
    await startProcessing(el, async () => await run(engine, datasetPath, sample));
    
    if (!runAllProcessing) {
        await plotLinearRegression();
    }

    stopProcess(el);
};

// Attach event listeners to all buttons with class "lr-wasm"
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll("button.lr-wasm").forEach(button => {
        button.addEventListener("click", async () => {
            await handleLinearRegressionRustWasm(button);
        });
    });
});

export default handleLinearRegressionRustWasm;
