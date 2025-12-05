from config import Config

# Set new model configuration
Config.set_model_config(
    embedding_key="bge-small",
    llm_key="mistral"
)

print("✓ Configuration updated!")
print(f"Embedding: {Config.EMBEDDING_PROVIDER}/{Config.EMBEDDING_MODEL}")
print(f"LLM: {Config.LLM_PROVIDER}/{Config.LLM_MODEL}")