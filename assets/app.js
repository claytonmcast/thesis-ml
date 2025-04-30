// Function to sleep for a specified time (in milliseconds)
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Global variables
var currentResultItem = null; // Holds the current result item
var executionTries = 0; // Keeps track of execution tries
var runAllProcessing = false; // Flag to control whether all processing should run

// Function to start processing with retries and updates
async function startProcessing(el, process, position) {
    var tries = Number(document.getElementById("tries").value); // Get the number of tries from the input field
    var pinExecutionTries = executionTries; // Pin the current execution tries for consistency

    // Loop through execution tries
    for (let i = pinExecutionTries; i < tries; i++) {
        executionTries = i + 1; // Update execution tries
        el.querySelector('.process-number').innerHTML = (tries + 1 - executionTries); // Update the process number on UI

        // Wait for process to complete and sleep for 1 second
        await process();
        await sleep(1000); // Wait for 1 second (1000 ms)
    }

    executionTries = 0; // Reset execution tries after completion
    var end = new Date(); // Record the end time

    // If not running all processing, update the result item
    if (runAllProcessing != true) {
        await updateResultItem({
            end,
            result_item_id: currentResultItem.id
        });
    }
}

// Function to plot the linear regression results
async function plotLinearRegression() {
    var tries = Number(document.getElementById("tries").value); // Get number of tries
    const response = await fetch("/api/plot_linear_regression?id=" + currentResultItem.id + "&tries=" + tries);
    const data = await response.json(); // Parse the JSON response

    document.querySelector('#refresh-result-grid-component').click();
    return data;
}

// Function to plot the neural network results
async function plotNeuralNetwork() {
    var tries = Number(document.getElementById("tries").value); // Get number of tries
    const response = await fetch("/api/plot_neural_network?id=" + currentResultItem.id + "&tries=" + tries);
    const data = await response.json(); // Parse the JSON response
    return data;
}

// Function to get the result list
async function getResultList() {
    const response = await fetch("/result_list.json");
    const data = await response.json(); // Parse the JSON response
    return data;
}

// Function to get the confidence interval data from a given location
async function getConfidenceInterval(location) {
    const response = await fetch(location);
    const data = await response.json(); // Parse the JSON response
    return data;
}

// Function to get a new result item
async function getNewResultItem(is_run_all) {
    var tries = Number(document.getElementById("tries").value); // Get number of tries
    const response = await fetch("/api/new_result_item?tries=" + tries + "&isRunAll=" + (is_run_all == true) + "&start=" + (new Date()).toISOString());
    const data = await response.json(); // Parse the JSON response

    // Store the result item and trigger UI refresh
    currentResultItem = data;
    document.querySelector('#refresh-result-grid-component').click();

    return data;
}

// Function to update the result item on the server
async function updateResultItem(resultItem) {
    const response = await fetch("/api/update_result_item", {
        method: "POST", // Specifies the request method
        headers: {
            "Content-Type": "application/json" // Tells the server the request body is JSON
        },
        body: JSON.stringify(resultItem) // Convert the data object to JSON string
    });

    const responseData = await response.json(); // Parse the response as JSON
    document.querySelector('#refresh-result-grid-component').click(); // Refresh the result grid
    return responseData;
}

// Function to append a new experiment to the server
async function appendExperiment(experiment) {
    const response = await fetch("/api/append_experiment", {
        method: "POST", // Specifies the request method
        headers: {
            "Content-Type": "application/json" // Tells the server the request body is JSON
        },
        body: JSON.stringify(experiment) // Convert the data object to JSON string
    });

    const responseData = await response.json(); // Parse the response as JSON
    document.querySelector('#refresh-result-grid-component').click(); // Refresh the result grid
    return responseData;
}

// Function to start processing and disable buttons
function startProcess(el) {
    disable_enable_buttons(true); // Disable buttons
    el.classList.add("processing"); // Add processing class to element
}

// Function to stop processing and enable buttons
function stopProcess(el) {
    disable_enable_buttons(false); // Enable buttons
    el.classList.remove("processing"); // Remove processing class from element
    document.querySelector('#refresh-result-grid-component').click(); // Refresh the result grid
}

// Function to enable/disable buttons
function disable_enable_buttons(isDisabled) {
    document.querySelectorAll('button').forEach(element => {
        element.disabled = isDisabled; // Disable or enable button
        if (!isDisabled) {
            element.classList.remove("processing"); // Remove processing class
        }
    });
    document.getElementById('tries').disabled = isDisabled; // Disable tries input field
}

function overleafOutputLR(data){
    let formattedTable = "";
    data.forEach(row=>{
        if(row.mse == undefined || row.mse.ci_lower == undefined){
            debugger;
        }
        var format = `
${row.platform.replaceAll(' ', '\\_')} & ${row.dataset_size.replace('%', '\\%')} & 
\\begin{tabular}[t]{@{}c@{}} ${row.training_time.ci_lower} \\\\ $< \\mu <$ \\\\ ${row.training_time.ci_upper} \\end{tabular} & 
\\begin{tabular}[t]{@{}c@{}} ${row.inference_time.ci_lower} \\\\ $< \\mu <$ \\\\ ${row.inference_time.ci_upper} \\end{tabular} & 
\\begin{tabular}[t]{@{}c@{}} ${row.mse.ci_lower} \\\\ $< \\mu <$ \\\\ ${row.mse.ci_upper} \\end{tabular} & 
\\begin{tabular}[t]{@{}c@{}} ${row.r2.ci_lower} \\\\ $< \\mu <$ \\\\ ${row.r2.ci_upper} \\end{tabular} \\\\ \\hline
        `;
        formattedTable += format;
    })
    window.ci_overleaf_lr = formattedTable;
    return formattedTable;
}

function overleafOutputNN(data){
    let formattedTable = "";
    data.forEach(row=>{

        var format = `
${row.platform.replaceAll(' ', '\\_')} & ${row.dataset_size.replace('%', '\\%')}\\% & 
\\begin{tabular}[t]{@{}c@{}} ${row.training_time.ci_lower} \\\\ $< \\mu <$ \\\\ ${row.training_time.ci_upper} \\end{tabular} & 
\\begin{tabular}[t]{@{}c@{}} ${row.inference_time.ci_lower} \\\\ $< \\mu <$ \\\\ ${row.inference_time.ci_upper} \\end{tabular} & 
\\begin{tabular}[t]{@{}c@{}} ${row.accuracy.ci_lower} \\\\ $< \\mu <$ \\\\ ${row.accuracy.ci_upper} \\end{tabular} & 
\\begin{tabular}[t]{@{}c@{}} ${row.loss.ci_lower} \\\\ $< \\mu <$ \\\\ ${row.loss.ci_upper} \\end{tabular} \\\\ \\hline
        `;
        formattedTable += format;
    })

    window.ci_overleaf_nn = formattedTable;
    return formattedTable;
}


// Function to transform and sort the data for the confidence interval object
function transformData(data) {
    const platformOrder = ["tensorflow_js_cpu", "tensorflow_js_webgpu", "tensorflow_js_wasm", "rust_wasm_cpu", "python_gpu"];
    const datasetOrder = ['10%', '50%', '100%'];

    let result = [];

    // Iterate over datasetOrder to process data for each dataset
    datasetOrder.forEach(datasetSize => {
        // Iterate over platformOrder to process data for each platform
        platformOrder.forEach(platform => {
            if (data[platform]) {
                if (data[platform][datasetSize]) {
                    let entry = {
                        platform: platform.replace(/_/g, ' '), // Format platform name for readability
                        dataset_size: datasetSize,
                        training_time: data[platform][datasetSize].training_time,
                        inference_time: data[platform][datasetSize].inference_time
                    };

                    // Add additional data fields if available
                    if (data[platform][datasetSize].mse) entry["mse"] = data[platform][datasetSize].mse;
                    if (data[platform][datasetSize].r2) entry["r2"] = data[platform][datasetSize].r2;
                    if (data[platform][datasetSize].accuracy) entry["accuracy"] = data[platform][datasetSize].accuracy;
                    if (data[platform][datasetSize].loss) entry["loss"] = data[platform][datasetSize].loss;

                    result.push(entry); // Push the entry to the result array
                }
            }
        });
    });

    return result;
}
