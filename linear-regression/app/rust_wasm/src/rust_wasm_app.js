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

async function run(engine, datasetPath) {  
     await init();  // Initialize the WebAssembly module
    // Fetch the dataset from the API
    const { features, target } = await fetchDataset(datasetPath, "price", "area");
    const features1D = features.flat(); 
    const target1D = target.flat(); 
    const data = await fetch_and_train(features1D, target1D); 
    console.log(data);  // Log the prediction result
    
    downloadJson(data, "rust_wasm_cpu_" + datasetPath.replace("house_price/", "").replace('.csv', ''));
}

document.addEventListener('DOMContentLoaded', (event) => {
    Array.from(document.querySelectorAll("button.lr-wasm")).forEach(el => {
        el.addEventListener("click", async () => {
            var dsPath = el.getAttribute('dataset');  
            var engine = el.getAttribute('engine');  
            startProcess(el);
            let result = await run(engine, dsPath);
            stopProcess(el);
            console.log(result);
        });
    });
});