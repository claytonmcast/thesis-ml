import init, { fetch_and_train } from "../pkg/rust_wasm.js"; 

// Fetch the binary data for the dataset from the server
async function fetchNabDataset(datasetPath) {
    // Fetch the dataset using the provided path
    const response = await fetch('neural_network/datasets/nab/' + datasetPath);
    const arrayBuffer = await response.arrayBuffer(); // Get raw binary data from the response
    const bytes = new Uint8Array(arrayBuffer); // Convert to Uint8Array for easier manipulation
    
    // Log the length and first few bytes for debugging purposes
    //console.log("Fetched bytes length:", bytes.length);
   // console.log("First few bytes:", bytes.slice(0, 10)); 

    return bytes;
}

// Main function to fetch the dataset and train the model
async function runNab(trainingPercentage, sample) {
    const startProcessTs = new Date(); // Record start time of the process
    await init(); // Initialize the WebAssembly module

    // Fetch the training and test datasets
    const trainImages = await fetchNabDataset('/mnist_images.nab');
    const trainLabels = await fetchNabDataset('/mnist_labels.nab');
   // console.log(trainImages); // Log the fetched training images

    // Call the WebAssembly function to train the model
    try {
        const result = await fetch_and_train(trainImages, trainLabels, trainingPercentage); // Fetch training results
        const endProcessTs = new Date(); // Record end time of the process

        // Prepare the path for storing the training results
        const experiments_path = "neural_network/training_result/" + currentResultItem.id;

        // Append the experiment results to the experiment history
        await appendExperiment({
            experiment: {
                try: executionTries,
                type: "Neural Network Rust WASM CPU",
                sample,
                title: "Neural Network Rust WASM CPU " + sample,
                start: startProcessTs,
                end: endProcessTs,
                platform: 'rust_wasm_cpu',
                result_item_id: currentResultItem.id,
                location: experiments_path,
                try_path: experiments_path + "/" + executionTries,
                experiment_path: experiments_path + "/" + executionTries + '/rust_wasm_cpu',
                result_path:  experiments_path + "/" + executionTries + '/rust_wasm_cpu/nn_mnist_rust_wasm_cpu_sample_' + sample + '.json'
            },
            results: result // Store the result of the training
        });  
    } catch (error) {
        // Catch and log any errors during the training process
        console.error("Error during prediction:", error);
    }
}

// Handle click events for training the neural network
const handleNeuralNetworkRustWasm = async (el, position) => {
    // Extract attributes from the clicked element
    var trainingPercentage = el.getAttribute('dataset');  
    var engine = el.getAttribute('engine');  
    var sample = el.getAttribute('sample');

    // If all processing is not active, fetch a new result item
    if(runAllProcessing != true){
        await getNewResultItem();
    }

    startProcess(el); // Start the process indicator
    await startProcessing(el, async () => await runNab(parseFloat(trainingPercentage), sample), position); // Run the training
    // If all processing is not active, plot the results
    if(runAllProcessing != true){
        await plotNeuralNetwork();
    }
    stopProcess(el); // Stop the process indicator
}

// Add event listeners for buttons to trigger the neural network training
document.addEventListener('DOMContentLoaded', (event) => {
    Array.from(document.querySelectorAll("button.mnist-wasm")).forEach(el => {
        el.addEventListener("click", async () => {
            await handleNeuralNetworkRustWasm(el); // Handle button click
        });
    });
});

// Export the handle function for use in other modules
export default handleNeuralNetworkRustWasm;
