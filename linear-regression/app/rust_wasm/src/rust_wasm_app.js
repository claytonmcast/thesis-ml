import init, { fetch_and_train } from "../pkg/rust_wasm.js"; 
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

async function run(engine, datasetPath, sample) {  
    const startProcessTs = new Date();
     await init();  // Initialize the WebAssembly module
    // Fetch the dataset from the API
    const { features, target } = await fetchDataset(datasetPath, "price", "area");
    const features1D = features.flat(); 
    const target1D = target.flat(); 
    const data = await fetch_and_train(features1D, target1D);  
    const endProcessTs = new Date();
    

    const experiments_path = "linear-regression/training_result/" + currentResultItem.id;
    appendExperiment({
        experiment: {
            try: executionTries,
            type: "Linear Regression Rust WASM CPU",
            sample,
            title: "Linear Regression Rust WASM CPU " + sample,
            start: startProcessTs,
            end: endProcessTs,
            result_item_id: currentResultItem.id,
            location: experiments_path,
            experiment_path: experiments_path + "/" + executionTries,
            result_path:  experiments_path + "/" + executionTries + '/rust_wasm_cpu/' + "rust_wasm_cpu_" + datasetPath.replace("house_price/", "").replace('.csv', '') + '.json'
        },
        results: data
    })   
} 

const handleLinearRegressionRustWasm = async (el) =>{
    var dsPath = el.getAttribute('dataset');  
    var engine = el.getAttribute('engine');  
    var sample = el.getAttribute('sample');

    if(runAllProcessing != true){
        getNewResultItem();
    }
     
    startProcess(el);
    await startProcessing(el, async ()=> await run(engine, dsPath, sample));
    stopProcess(el);
   
}
 
document.addEventListener('DOMContentLoaded', (event) => {
    Array.from(document.querySelectorAll("button.lr-wasm")).forEach(el => {
        el.addEventListener("click", async () => {
            await handleLinearRegressionRustWasm(el);
        });
    });
});

export default handleLinearRegressionRustWasm;