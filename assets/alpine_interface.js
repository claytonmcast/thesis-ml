// Function to bind and initialize the result list table
function bindResultListTable() {
    return {
        data: [],             // Holds the data for the result list
        initialized: false,   // Flag to check if the component has been initialized

        // Initialize the component
        async init() { 

            // Fetch the result list from the server
            const resultList = await getResultList();

            // If results are available
            if(runAllProcessing != true && resultList.length > 0){ 
                var resultItem = resultList[resultList.length - 1];
                var isRunAll = resultItem.isRunAll == 'true';  // Check if 'Run All' was executed
                
                // If the 'end' property is missing, the experiment needs to continue
                if(resultItem.end == undefined){
                    document.getElementById("tries").value = resultItem.tries;
                    currentResultItem = resultItem; // Set the current result item
                    var experiments = resultItem.experiments;

                    // If it's a 'Run All' experiment, trigger the runAll function
                    if(isRunAll) {
                        var funcRun = false;
                        do {
                            if (typeof window.runAll === "function") { 
                                window.runAll(experiments.length);
                                funcRun = true;  // Mark the function as run
                            }
                            await sleep(1000);  // Wait before trying again
                        } while(!funcRun);
                    }
                }
            }

            // Group experiments by 'try' and sort them by 'sample' value
            resultList.forEach(result => {
                const groupedExperiments = result.experiments.reduce((acc, exp) => {
                    const key = exp.try; // Use 'try' as the key for grouping
                    if (!acc[key]) acc[key] = [];
                    acc[key].push(exp);
                    return acc;
                }, {});

                // Convert grouped experiments to a sorted array
                result.experiments = Object.entries(groupedExperiments)
                    .map(([tryNumber, experiments]) => ({
                        try: Number(tryNumber),  // Convert 'try' to a number
                        experiments: experiments.sort((a, b) => Number(a.sample.replace('%', '')) - Number(b.sample.replace('%', '')))
                    }))
                    .sort((a, b) => a.try - b.try)  // Sort by 'try' number
                    .reverse();  // Reverse the order to show the latest 'try' first

                // Initialize arrays for confidence intervals
                result.lr_confidence_interval = [];
                result.nn_confidence_interval = [];
            });

            this.data = resultList;  // Set the data for the result grid
        },

        // Toggle the display of experiments for a given operation
        displayExperiments(operation) {
            operation.display_experiments = !operation.display_experiments;
            console.log(this.data);  // Log the updated data (for debugging)
        },

        // Display metrics for the given operation
        async displayMetrics(operation) {
            operation.show_metric = !operation.show_metric;  // Toggle the metric display flag
            var firstExperiment = operation.experiments[0].experiments[0];  // First experiment
            var firstLocation = firstExperiment.location + '/confidence_interval_metric.json'; // Location for the first experiment's metric
            
            // If it's a Linear Regression experiment, fetch the confidence interval
            if(firstExperiment.type.indexOf('L') === 0){
                var ci = await getConfidenceInterval(firstLocation);
                operation.lr_confidence_interval = transformData(ci);  // Transform and add the data
                operation.show_metric_lr = true;  // Show LR metric
                overleafOutputLR(operation.lr_confidence_interval)
            }

            // If it's a Neural Network experiment, fetch the confidence interval
            if(firstExperiment.type.indexOf('N') === 0){
                var ci = await getConfidenceInterval(firstLocation);
                operation.nn_confidence_interval = transformData(ci);  // Transform and add the data
                operation.show_metric_nn = true;  // Show NN metric
                overleafOutputNN(operation.nn_confidence_interval)
            }

            // Check the last experiment's type to handle any differences in metrics
            var lastExperiment = operation.experiments[0].experiments[operation.experiments[0].experiments.length - 1];
            var lastLocation = lastExperiment.location + '/confidence_interval_metric.json'; 

            // If the first and last experiments are of different types, fetch the appropriate metric
            if(firstExperiment.type.indexOf('L') !== lastExperiment.type.indexOf('L')){
                var ci = await getConfidenceInterval(lastLocation);
                operation.nn_confidence_interval = transformData(ci);  // Add the NN confidence interval
                operation.show_metric_nn = true;  // Show NN metric
                overleafOutputNN(operation.nn_confidence_interval)
            }
        }
    };
}
