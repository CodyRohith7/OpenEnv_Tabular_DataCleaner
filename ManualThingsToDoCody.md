# Manual Things To Do (Cody)

This document contains all the manual steps you must perform to successfully submit this project to the OpenEnv Hackathon. Please follow these carefully.

## 1. Prerequisites (Do this ASAP)
- **Hugging Face Account**: Ensure you are registered at [Hugging Face](https://huggingface.co/) and create a new **Space** (choose Docker as the backend). Name it something like `data-cleaning-env`.
- **Hugging Face CLI Login**: Open your local terminal and run:
  ```bash
  huggingface-cli login
  ```
  *Paste your HF Access Token when prompted (you can generate it in your HF Settings -> Access Tokens).*

## 2. Platform Application (April 1st)
When Round 1 opens on the Hackathon platform:
- Submit the **Application Form**.
- Select the problem statement related to "Data Cleaning" or "Real-world task simulation".

## 3. Configure API Credentials Locally
To test the environment locally with the baseline inference script, set your API keys as environment variables.
For Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="your_actual_openai_api_key_here"  # Or your groq/HF token if using a different base url
$env:MODEL_NAME="gpt-4o" # or whatever you have access to
```
*(If you are using a HuggingFace endpoint, make sure to set `API_BASE_URL` properly as per the docs).*

## 4. Run Pre-Submission Checklist
Always run the validation shell script locally to ensure everything builds and complies.
1. Run the local backend to test:
   ```bash
   uv run server  # Or start via your preferred docker run command
   ```
2. Test the baseline locally:
   ```bash
   python inference.py
   ```
3. Run the automated validator provided by the hackathon (with your newly created space URL):
   ```bash
   ./validate-submission.sh https://codyrohith7-data-cleaning-env.hf.space .
   ```

## 5. Deploy to Hugging Face
Once the environment is verified:
1. Push to your Hugging Face Space using the OpenEnv CLI:
   ```bash
   openenv push --repo-id CodyRohith7/data-cleaning-env
   ```
2. Monitor the "Build" logs in your Hugging Face Space UI to ensure the Docker container starts successfully.

## 6. Final Submission
- Once your Space is live (shows "Running" status on HF).
- Copy your Space URL (e.g., `https://huggingface.co/spaces/CodyRohith7/data-cleaning-env`).
- Go to the Hackathon submission page and **Paste your HF Spaces URL** before the deadline (**8 April 2026, 11:59 PM IST**).
