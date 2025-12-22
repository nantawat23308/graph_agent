from pymilvus import Collection, MilvusException, connections, db, utility
from src.langchain_milvus.constant import URI, DB_NAME


conn = connections.connect(host="127.0.0.1", port=19530)

# Check if the database exists
try:
    existing_databases = db.list_database()
    if DB_NAME in existing_databases:
        print(f"Database '{DB_NAME}' already exists.")

        # Use the database context
        db.using_database(DB_NAME)

        # Drop all collections in the database
        collections = utility.list_collections()
        for collection_name in collections:
            collection = Collection(name=collection_name)
            collection.drop()
            print(f"Collection '{collection_name}' has been dropped.")

        db.drop_database(DB_NAME)
        print(f"Database '{DB_NAME}' has been deleted.")
    else:
        print(f"Database '{DB_NAME}' does not exist.")
        database = db.create_database(DB_NAME)
        print(f"Database '{DB_NAME}' created successfully.")
except MilvusException as e:
    print(f"An error occurred: {e}")
