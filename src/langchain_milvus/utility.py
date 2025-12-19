from langchain_aws import BedrockEmbeddings
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()


def get_bedrock_embeddings() -> Embeddings:
    """Get Bedrock embeddings instance."""
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-west-2",
    )
    return embeddings
