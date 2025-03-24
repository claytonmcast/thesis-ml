import init, { fetch_and_train } from "../pkg/rust_wasm.js"; 
 
async function fetchNabDataset(datasetPath) {
    const response = await fetch('neural-network/datasets/nab/' + datasetPath);
    const arrayBuffer = await response.arrayBuffer(); // Get raw binary data
    const bytes = new Uint8Array(arrayBuffer); // Convert to Uint8Array
    
    console.log("Fetched bytes length:", bytes.length);
    console.log("First few bytes:", bytes.slice(0, 10)); 

    return bytes;
}

async function runNab(trainingPercentage) {
    await init(); // Initialize the WebAssembly module

    // Fetch the training and test datasets
    const trainImages = await fetchNabDataset('/mnist_images.nab');
    const trainLabels = await fetchNabDataset('/mnist_labels.nab');
    console.log(trainImages);

    // Call the WebAssembly function
    try {
        const result = await fetch_and_train(trainImages, trainLabels, trainingPercentage);
        
        downloadJson(result, 'nn_mnist_rust_wasm_cpu_sample_' + (trainingPercentage * 100) + "%");
        console.log(result);
    } catch (error) {
        console.error("Error during prediction:", error);
    }
}

document.addEventListener('DOMContentLoaded', (event) => {
    Array.from(document.querySelectorAll("button.mnist-wasm")).forEach(el => {
        el.addEventListener("click", async () => {
            var trainingPercentage = el.getAttribute('dataset');  
            var engine = el.getAttribute('engine');  
            startProcess(el);
            await runNab(parseFloat(trainingPercentage));
            stopProcess(el);
           
        });
    });
});