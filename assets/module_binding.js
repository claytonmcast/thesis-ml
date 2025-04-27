// Importing handlers for different machine learning processes
import handleLinearRegression from '/linear_regression/app/tensorflow_js/tensorflow_js_app.js';
import handleLinearRegressionRustWasm from '/linear_regression/app/rust_wasm/src/rust_wasm_app.js';
import handleNeuralNetwork from '/neural_network/app/tensorflow_js/tensorflow_js_app.js';
import handleNeuralNetworkRustWasm from '/neural_network/app/rust_wasm/src/rust-wasm-app.js';

// Event listener for when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', (event) => {
    // Adding event listener to "Run All" button
    var el = document.getElementById('run-all');
    el.addEventListener("click", async () => {
        await runAll();
    });
});

// Function to plot all necessary graphs
async function plotAll() {
    const end = new Date();
    // Update result item after processing
    await updateResultItem({
        end: end,
        result_item_id: currentResultItem.id
    });
    await plotLinearRegression();
    await plotNeuralNetwork();
    
    document.querySelector('#refresh-result-grid-component').click(); // Refresh the result grid
}

// Main function to run all experiments sequentially
async function runAll(lastExecutedExperiment) {
    // Prevent running if already processing
    if (runAllProcessing == true) {
        return;
    }

    runAllProcessing = true;
    const tries = Number(document.getElementById("tries").value);
    const totalButtons = 30;  // Total number of button types to execute

    const totalExecutions = totalButtons * tries;

    // If no last executed experiment, initialize a new result item
    if (lastExecutedExperiment === undefined) {
        await getNewResultItem(true);
        lastExecutedExperiment = 0;
    }

    // If all experiments are done, stop processing
    if (lastExecutedExperiment >= totalExecutions) {
        console.log("All done!");
        startProcess(document.getElementById('run-all'));
        await plotAll();
        runAllProcessing = false;
        stopProcess(document.getElementById('run-all'));
        return;
    }

    const buttonIndex = Math.floor(lastExecutedExperiment / tries);

    // Collect all button elements in the same order as your totalButtons count (30)
    const allButtons = [
        ...document.querySelectorAll("button.lr-tf"),
        ...document.querySelectorAll("button.lr-wasm"),
        ...document.querySelectorAll("button.lr-python"),
        ...document.querySelectorAll("button.mnist-tf"),
        ...document.querySelectorAll("button.mnist-wasm"),
        ...document.querySelectorAll("button.mnist-python")
    ];

    // Safeguard check: ensure index is within bounds
    if (buttonIndex >= allButtons.length) {
        console.error("Index out of range: nextIndex is too large.");
        return;
    }

    const executionIndexForThisButton = lastExecutedExperiment % tries;
    executionTries = executionIndexForThisButton;

    try {
        // Loop over the button elements and process accordingly
        for (let i = buttonIndex; i < allButtons.length; i++) {
            const button = allButtons[i];

            if (button.classList.contains("lr-tf")) {
                await handleLinearRegression(button, executionIndexForThisButton);
            } else if (button.classList.contains("lr-wasm")) {
                await handleLinearRegressionRustWasm(button, executionIndexForThisButton);
            } else if (button.classList.contains("lr-python")) {
                await handleLinearRegressionPython(button, executionIndexForThisButton);
            } else if (button.classList.contains("mnist-tf")) {
                await handleNeuralNetwork(button, executionIndexForThisButton);
            } else if (button.classList.contains("mnist-wasm")) {
                await handleNeuralNetworkRustWasm(button, executionIndexForThisButton);
            } else if (button.classList.contains("mnist-python")) {
                await handleNeuralNetworkPython(button, executionIndexForThisButton);
            }
        }
    } catch (ex) {
        // In case of error, redirect to a blank page
        location.href = 'blank.html?dt=' + new Date();
        return;
    }

    await plotAll();
    runAllProcessing = false;
}

// Expose the runAll function to global scope
window.runAll = runAll;

// Function to run Python code via API call
async function runPython(type, dataset, sample) {
    var tries = Number(document.getElementById("tries").value);
    const response = await fetch(`/api/run_python?type=${type}&try=${executionTries}&sample=${sample}&dataset=${dataset}&result_item_id=${currentResultItem.id}`);
    const data = await response.json();
}

// Function to handle Linear Regression with Python
async function handleLinearRegressionPython(el, position) {
    var dataset = el.getAttribute('dataset');
    var engine = el.getAttribute('engine');
    var sample = el.getAttribute('sample');

    startProcess(el);

    if (runAllProcessing != true) {
        await getNewResultItem();
    }

    // Start processing and run Python Linear Regression
    await startProcessing(el, async () => await runPython("Linear Regression Python GPU", dataset, sample), position);

    if (runAllProcessing != true) {
        await plotLinearRegression();
    }

    stopProcess(el);
}

// Function to handle Neural Network with Python
async function handleNeuralNetworkPython(el, position) {
    var dataset = el.getAttribute('dataset');
    var engine = el.getAttribute('engine');
    var sample = el.getAttribute('sample');

    startProcess(el);

    if (runAllProcessing != true) {
        await getNewResultItem();
    }

    // Start processing and run Neural Network with Python
    await startProcessing(el, async () => await runPython("Neural Network Python GPU", dataset, sample));

    if (runAllProcessing != true) {
        await plotNeuralNetwork();
    }

    stopProcess(el);
}

// Adding event listeners for Linear Regression and Neural Network Python buttons
document.addEventListener('DOMContentLoaded', (event) => {
    Array.from(document.querySelectorAll("button.lr-python")).forEach(el => {
        el.addEventListener("click", async () => {
            await handleLinearRegressionPython(el);
        });
    });
    Array.from(document.querySelectorAll("button.mnist-python")).forEach(el => {
        el.addEventListener("click", async () => {
            await handleNeuralNetworkPython(el);
        });
    });
});
