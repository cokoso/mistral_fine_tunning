# mistral_fine_tunning
sample Jupyter notebook for fine-tuning Mistral LLM in Azure AI Studio, along with sample JSON and CSV dataset files.

## 1. **Jupyter Notebook** - Fine-tune Mistral LLM with Transformers
This notebook includes:
- Loading your training data from JSON or CSV files
- Automatic 10% train-validation split
- Tokenization with proper padding and truncation
- Model loading with GPU optimization (fp16, device_map="auto")
- Complete training pipeline with evaluation
- Inference testing on the fine-tuned model
- Model saving for deployment

**Key features:**
- Uses `transformers` library with Mistral-7B
- Configurable batch size, learning rate, and epochs
- Gradient accumulation for better convergence
- Built-in validation during training
- Evaluation metrics on validation set

## 2. **Sample JSON Dataset**
10 instruction-output pairs covering AI/ML concepts. Format: `{"instruction": "...", "output": "..."}`

## 3. **Sample CSV Dataset**
Same data in CSV format with `instruction` and `output` columns.

### To use in Azure AI Studio:

1. **Upload the notebook** to your Azure AI Studio environment
2. **Prepare your data**:
   - Use either JSON (list of objects) or CSV format
   - Each row should have instruction-output pairs or text content
3. **Update the path**: Change `TRAIN_FILE` to point to your actual dataset
4. **Adjust parameters** (batch size, learning rate, epochs) based on your hardware
5. **Run the notebook** - it will handle the train-validation split automatically

The notebook will save your fine-tuned model to `./mistral_finetuned/` which you can then download or deploy directly from Azure AI Studio.
