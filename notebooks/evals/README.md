# Calculator Tool Evaluation

This directory contains a Promptfoo-based evaluation harness for a tool-calling agent. The agent uses the Azure OpenAI Responses API and a local calculator tool to answer math tasks, then Promptfoo checks each answer against the expected output.

> 💡 Promptfoo is an evaluation runner for LLM apps. It manages prompts, providers, assertions, result exports, and reports so this project does not need a custom evaluation loop.

## Files

```text
notebooks/evals/
├── evaluation.xml                # Source math tasks and expected responses
├── promptfooconfig.json          # Promptfoo config generated from the XML tasks
├── promptfoo_agent_provider.py   # Python provider with the agent loop and calculator tool
├── tool-evaluation.ipynb         # Notebook for generating config, running evals, and reporting
└── README.md
```

## Flow

```text
evaluation.xml
    |
    v
promptfooconfig.json
    |
    v
npx promptfoo eval
    |
    v
promptfoo_agent_provider.py
    |
    v
Azure Responses API + calculator tool
    |
    v
promptfoo-results.jsonl
```

## Prerequisites

1. Install Python dependencies from the repository root:

   ```bash
   uv sync
   ```

2. Make sure `npx` is available. Promptfoo is run with `npx --yes promptfoo@latest`, so no JavaScript package file is required.

3. Create a repository-root `.env` file with your Azure OpenAI key:

   ```bash
   AZURE_OPENAI_API_KEY=your-key-here
   ```

   Optional overrides:

   ```bash
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
   AZURE_OPENAI_API_VERSION=2025-04-01-preview
   AZURE_OPENAI_MODEL=gpt-5.5
   ```

## Run from the command line

From this directory:

```bash
cd notebooks/evals
npx --yes promptfoo@latest eval \
  --config promptfooconfig.json \
  --output promptfoo-results.jsonl \
  --no-cache \
  --env-file ../../.env
```

Promptfoo writes `promptfoo-results.jsonl`. Result files are ignored by Git via `notebooks/evals/promptfoo-results.*`.

## Run from the notebook

Open `tool-evaluation.ipynb` and run the cells top to bottom. The notebook:

1. Loads tasks from `evaluation.xml`.
2. Rebuilds `promptfooconfig.json`.
3. Runs Promptfoo in this directory.
4. Reads `promptfoo-results.jsonl`.
5. Builds a compact Markdown report.

## Provider details

`promptfoo_agent_provider.py` exposes `call_api(prompt, options, context)`, which Promptfoo calls for each test case. The provider:

- Sends the task to Azure OpenAI with the evaluator system prompt.
- Exposes one local `calculator` function tool.
- Tracks tool call counts and durations.
- Extracts the final `<response>` tag as the value Promptfoo asserts against.
- Stores summary, feedback, raw output, and tool metrics in Promptfoo metadata.

The calculator uses a small safe AST interpreter instead of raw `eval`. It supports arithmetic, common math functions like `sqrt` and `log10`, and constants like `pi`.
