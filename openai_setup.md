# OpenAI API Setup Guide

## Quick Setup

1. **Get your OpenAI API key:**
   - Go to https://platform.openai.com/account/api-keys
   - Create a new API key
   - Copy the key (it starts with `sk-...`)

2. **Set your API key in the .env file:**
   ```bash
   # Open the .env file
   nano .env
   
   # Replace 'your_openai_api_key_here' with your actual API key
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. **Alternative: Set environment variable directly:**
   ```bash
   export OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

4. **Run the benchmark:**
   ```bash
   python benchmark/run_benchmark.py
   ```

## Complete .env file example:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-abc123...your-key-here

# HuggingFace Configuration (optional)
HUGGINGFACE_API_TOKEN=hf_your_token_here

# Groq Configuration (for future use)
GROQ_API_KEY=your_groq_api_key_here

# Other configurations
PYTHONPATH=.
```

## Security Notes:

- **Never commit your .env file to Git** (it's in .gitignore)
- Keep your API keys secure and private
- Monitor your OpenAI usage at https://platform.openai.com/usage

## Troubleshooting:

If you get "No API key found" errors:
1. Check that your .env file is in the project root
2. Verify the API key format (starts with `sk-`)
3. Restart your terminal after setting environment variables
4. Make sure there are no spaces around the `=` in your .env file

## Cost Information:

The benchmark uses GPT-4o-mini which costs approximately:
- $0.000150 per 1K input tokens (10x cheaper than GPT-3.5-turbo)
- $0.000600 per 1K output tokens (3x cheaper than GPT-3.5-turbo)

The full benchmark should cost less than $0.10 to run.