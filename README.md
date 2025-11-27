# CSULB-RAGBot
 A LLM able to be ran locally on your own machine and help you by answering your university related questions  by fetching the latest data


# FLOW

- Take input question, retrieve using k=5 from chromadb, map the blocks to the website they were retrieved from.
- Get the latest data from the website, and use it to re-index those blocks, and delete the older ones
- Pass the data retrieved into context, and try to answer the question by supplementing from context.
- For any follow up questions, use only the context present unless some other data is needed