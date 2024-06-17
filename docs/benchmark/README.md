# Benchmarking :balance_scale:

We conducted a comprehensive benchmarking and analysis of a diverse range of state-of-the-art models from different strands of research. We mainly assessed these models using NAME and CRPS metrics across multiple forecasting horizons, repeating each experiment five times with different seeds to ensure result reliability.

Results of **time series foundation models** see [HERE](./FOUNDATION_MODEL.md).

## Long-term Forecasting Benchmarking 

Detailed configuration files can be found in folder [config/ltsf/](../../config/ltsf/).

Table 1. Results ($\textrm{mean}_{\textrm{std}}$) on long-term forecasting scenarios with the best in $\textbf{bold}$ and the second $\underline{\textrm{underlined}}$, each containing five independent runs with different seeds. The input sequence length is set to 36 for the ILI-L dataset and 96 for the others. Due to the excessive time and memory consumption of CSDI in producing long-term forecasts, its results are unavailable in some datasets.

![long-term forecasting experimental results](./figs/long_bench.jpg)



## Short-term Forecasting Benchmarking 

Detailed configuration files can be found in folder [config/stsf/](../../config/stsf/).

Table 2.Results ($\textrm{mean}_{\textrm{std}}$) on short-term forecasting scenarios with the best in $\textbf{bold}$ and the second $\underline{\textrm{underlined}}$, each containing five independent runs with different seeds.

![short-term forecasting experimental results](./figs/short_bench.jpg)
