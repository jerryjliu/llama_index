from llama_index.readers.chatgpt_conversations import (
    ChatGPTConversationsReader,
    ChatGPTMessageNodeParser,
)

# Specify the path to your conversations.json file
conversations_file = "path/to/your/conversations.json"  # Update this path

# Initialize the ChatGPTConversationsReader
reader = ChatGPTConversationsReader(input_file=conversations_file)

# Load conversations as Documents
documents = reader.load_data()
print(f"Number of documents loaded: {len(documents)}")

# Initialize the ChatGPTMessageNodeParser
parser = ChatGPTMessageNodeParser()

# Parse documents into nodes
nodes = parser(documents)
print(f"Number of nodes parsed: {len(nodes)}")

# Optionally, print out some information about the first few nodes
for i, node in enumerate(nodes[:5]):
    print(f"\nNode {i + 1}:")
    print(f"Text: {node.text}")
    print(f"Metadata: {node.metadata}")
