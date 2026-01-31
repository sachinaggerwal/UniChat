from config import Config

# Set new model configuration
Config.set_model_config(
    embedding_key="text-embedding-3-small",
    llm_key="gpt-4o-mini"
)

print("✓ Configuration updated!")
print(f"Embedding: {Config.EMBEDDING_PROVIDER}/{Config.EMBEDDING_MODEL}")
print(f"LLM: {Config.LLM_PROVIDER}/{Config.LLM_MODEL}")