#!/bin/sh

echo "NOTE! By downloading these checkpoints, you agree to the licenses of the original models and checkpoints."
echo ""
echo "- [Timer](https://github.com/thuml/Large-Time-Series-Model) created by thuml. The original model and its checkpoints are licensed under the MIT License. The checkpoints are distributed under the MIT License. You may not use these files except in compliance with the License. You may obtain a copy of the License at: https://github.com/thuml/Large-Time-Series-Model/blob/main/LICENSE."
echo "- [ForecastPFN](https://github.com/abacusai/ForecastPFN) created by abacusai. The original model and its checkpoints are licensed under the MIT License. The checkpoints are distributed under the Apache-2.0 License. You may not use these files except in compliance with the License. You may obtain a copy of the License at: https://github.com/abacusai/ForecastPFN/blob/main/LICENSE."
echo "- [UniTS](https://github.com/mims-harvard/UniTS) created by mims-harvard. The original model and its checkpoints are licensed under the MIT License. The checkpoints are distributed under the MIT License. You may not use these files except in compliance with the License. You may obtain a copy of the License at: https://github.com/mims-harvard/UniTS/blob/main/LICENSE."
echo "- [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama) created by time-series-foundation-models. The original model and its checkpoints are licensed under the MIT License. The checkpoints are distributed under the Apache-2.0 License. You may not use these files except in compliance with the License. You may obtain a copy of the License at: https://github.com/time-series-foundation-models/lag-llama/blob/main/LICENSE."
echo ""
echo "NOTE! By downloading these checkpoints, you agree to the licenses of the original models and checkpoints."
read -p "Do you want to continue? (yes/y to continue): " confirm

# Convert input to lowercase for comparison
confirm=$(echo "$confirm" | tr '[:upper:]' '[:lower:]')

if [ "$confirm" = "yes" ] || [ "$confirm" = "y" ]; then
    # Check if gdown is installed
    if pip show gdown > /dev/null 2>&1; then
        echo "gdown is already installed, skipping installation."
    else
        echo "gdown is not installed, installing..."
        pip install gdown
    fi
    # Download the folder
    gdown --folder 1FaCk9Lj9KZGEO09gehNqC4fbTj4wnN8j -O checkpoints
else
    echo "Download canceled."
fi