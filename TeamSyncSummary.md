# Project Overview: OpenEnv Tabular Data Cleaning Agent
**Collaborative Status Report for Hackathon Teammates**

## 1. The Core Idea (The "Why")
To build a high-utility, real-world RL environment for the OpenEnv Hackathon. Instead of a game, we've built a **Data Engineering Simulator**. It challenges AI agents to transform "messy" raw data into "clean" gold-standard data using declarative operations. This allows for rigorous evaluation of an agent's ability to reason over data schemas, normalization, and aggregation.

## 2. The Solution (The "What")
A containerized FastAPI environment that simulates a headless data cleaning workstation.
- **Environment**: `TabularDataCleaningEnv`
- **Domain**: Automated Data Engineering (Real-world Utility score: 30/30).
- **Interface**: Full OpenEnv spec compliance (Pydantic models for Actions, Observations, and Rewards).

## 3. The Architecture (The "How")
- **Backend (FastAPI)**: Serves `/reset`, `/step`, and `/state` endpoints. Hosted in a Docker container.
- **State Management**: Uses `pandas` internally to track transformations.
- **Action Space**: A fixed set of 10 operations (e.g., `DEDUP_ROWS`, `CONVERT_CURRENCY`, `GROUPBY_SUM`).
- **Reward Engine**: A deterministic grader that compares the current dataframe state against a hidden "Gold" CSV and assigns rewards based on `cell_accuracy` and `delta_accuracy`.
- **Inference Script (`inference.py`)**: A compliant baseline script that communicates with the model (GPT-4o/Qwen) and the environment, logging in the required `[START]`, `[STEP]`, `[END]` format.

## 4. Work Flow
1. **Reset**: Environment loads raw data (Easy, Medium, or Hard task).
2. **Observe**: Agent receives a JSON preview of the data and natural language instructions.
3. **Act**: Agent sends a JSON action (e.g., `op: NORMALIZE_CASE, column: city`).
4. **Reward**: Environment applies the Pandas transformation and calculates the improvement score.
5. **Done**: Episode ends when max steps are reached or data is 100% clean.

## 5. Things Completed ✅
- **OpenEnv Spec Compliance**: All typed models and state/step/reset methods implemented.
- **Multi-Tier Tasks**: Easy (single-col), Medium (deduplication), and Hard (aggregations) tasks designed.
- **Inference Logger**: Strict STDOUT formatting (2 decimal places, booleans, flush=True) finalized.
- **Dockerization**: `Dockerfile` optimized for Hugging Face Spaces (Port 7860).
- **Documentation**: Professional `README.md` with action/observation space definitions and setup guides.
- **Manual Checklist**: Created `ManualThingsToDoCody.md` for local testing and final submission steps.

## 6. Things To Be Done (Manual/Final Steps) ⏳
- **Local Validation**: Run the `validate-submission.sh` script to ensure it passes all 3/3 automated checks.
- **HF Space Creation**: Create the Hugging Face Space manually and get the API token.
- **Deployment**: Run `openenv push` to upload the code to Hugging Face.
- **Final Submission**: Paste the live HF Space URL into the hackathon portal before April 8, 11:59 PM IST.
- **Demo Record (Optional)**: If teammate wants to show visual progress, record the terminal output of `inference.py`.
