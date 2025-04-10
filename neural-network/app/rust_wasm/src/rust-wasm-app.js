import init, { fetch_and_train } from "../pkg/rust_wasm.js"; 
 
async function fetchNabDataset(datasetPath) {
    const response = await fetch('neural-network/datasets/nab/' + datasetPath);
    const arrayBuffer = await response.arrayBuffer(); // Get raw binary data
    const bytes = new Uint8Array(arrayBuffer); // Convert to Uint8Array
    
    console.log("Fetched bytes length:", bytes.length);
    console.log("First few bytes:", bytes.slice(0, 10)); 

    return bytes;
}

async function runNab(trainingPercentage, sample) {
    const startProcessTs = new Date();
    await init(); // Initialize the WebAssembly module

    // Fetch the training and test datasets
    const trainImages = await fetchNabDataset('/mnist_images.nab');
    const trainLabels = await fetchNabDataset('/mnist_labels.nab');
    console.log(trainImages);

    // Call the WebAssembly function
    try {
        const result = await fetch_and_train(trainImages, trainLabels, trainingPercentage);
        const endProcessTs = new Date();
        

        const experiments_path = "neural-network/training_result/" + currentResultItem.id;
        appendExperiment({
            experiment: {
                try: executionTries,
                type: "Neural Network Rust WASM CPU",
                sample,
                title: "Neural Network Rust WASM CPU " + sample,
                start: startProcessTs,
                end: endProcessTs,
                result_item_id: currentResultItem.id,
                location: experiments_path,
                experiment_path: experiments_path + "/" + executionTries,
                result_path:  experiments_path + "/" + executionTries + '/rust_wasm_cpu/nn_mnist_rust_wasm_cpu_sample_' + sample + '.json'
            },
            results: result
        })  
 
    } catch (error) {
        console.error("Error during prediction:", error);
    }
}

const handleNeuralNetworkRustWasm = async (el) =>{
    var trainingPercentage = el.getAttribute('dataset');  
    var engine = el.getAttribute('engine');  
    var sample = el.getAttribute('sample');

    if(runAllProcessing != true){
        getNewResultItem();
    }
    startProcess(el);
    await startProcessing(el, async ()=> await runNab(parseFloat(trainingPercentage), sample));
    stopProcess(el);
   
}

document.addEventListener('DOMContentLoaded', (event) => {
    Array.from(document.querySelectorAll("button.mnist-wasm")).forEach(el => {
        el.addEventListener("click", async () => {
            await handleNeuralNetworkRustWasm(el);
        });
    });
});

export default handleNeuralNetworkRustWasm;