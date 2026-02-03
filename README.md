# Databricks Multi-Agent RAG UI

Multi-agent RAG chat interface deployed as a Databricks App.

## Architecture

```
Gradio App → Agent Endpoints → Vector Search → Delta Tables ← PDFs (Volumes)
```

## Quick Start

### 1. Manual Setup (Databricks Workspace)

```sql
-- Create schema
CREATE SCHEMA IF NOT EXISTS catalog.rag_agents;

-- Create volumes for PDFs
CREATE VOLUME catalog.rag_agents.agent_a_pdfs;
CREATE VOLUME catalog.rag_agents.agent_b_pdfs;
CREATE VOLUME catalog.rag_agents.agent_c_pdfs;
```

Upload PDFs to respective volumes.

### 2. Run Data Pipeline

**Option A: Via Databricks Job (recommended)**

```bash
databricks bundle deploy -t dev \
  --var catalog=your_catalog \
  --var schema=rag_agents \
  --var warehouse_id=your_warehouse_id

# Run the job
databricks bundle run rag_data_pipeline -t dev
```

The job runs all notebooks in sequence with proper parameterization.

**Option B: Manual notebook execution**

Run notebooks in order, setting `catalog` and `schema` widget values:
1. `data_prep/01_parse_pdfs.ipynb` - Parse PDFs → Delta tables
2. `data_prep/02_create_indexes.ipynb` - Create Vector Search indexes
3. `agents/deploy_agents.ipynb` - Register models and deploy endpoints

### 3. Deploy App

```bash
# Set variables
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com

# Deploy
databricks bundle deploy -t dev \
  --var catalog=your_catalog \
  --var schema=rag_agents \
  --var warehouse_id=your_warehouse_id
```

## Configuration

Update these files before deployment:
- `agents/config_*.yml` - Vector search index names, LLM endpoint
- `src/chat_app/app.py` - AGENTS dict (display names → endpoint names)

Notebook parameters (`catalog`, `schema`) are passed via job or widget values.

## Structure

```
├── databricks.yml           # Asset bundle config
├── src/chat_app/
│   ├── app.py               # Gradio UI
│   └── requirements.txt
├── agents/
│   ├── agent.py             # RAG agent logic
│   ├── config_*.yml         # Per-agent configs
│   └── deploy_agents.ipynb  # Deployment notebook
└── data_prep/
    ├── 01_parse_pdfs.ipynb
    └── 02_create_indexes.ipynb
```
