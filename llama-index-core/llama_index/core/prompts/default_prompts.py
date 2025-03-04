"""Set of default prompts."""

from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

############################################
# Tree
############################################

DEFAULT_SUMMARY_PROMPT_TMPL = (
    "Write a summary of the following. Try to use only the "
    "information provided. "
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'SUMMARY:"""\n'
)

DEFAULT_SUMMARY_PROMPT = PromptTemplate(
    DEFAULT_SUMMARY_PROMPT_TMPL, prompt_type=PromptType.SUMMARY
)

# insert prompts
DEFAULT_INSERT_PROMPT_TMPL = (
    "Context information is below. It is provided in a numbered list "
    "(1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "---------------------\n"
    "Given the context information, here is a new piece of "
    "information: {new_chunk_text}\n"
    "Answer with the number corresponding to the summary that should be updated. "
    "The answer should be the number corresponding to the "
    "summary that is most relevant to the question.\n"
)
DEFAULT_INSERT_PROMPT = PromptTemplate(
    DEFAULT_INSERT_PROMPT_TMPL, prompt_type=PromptType.TREE_INSERT
)


# # single choice
DEFAULT_QUERY_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "the choice that is most relevant to the question: '{query_str}'\n"
    "Provide choice in the following format: 'ANSWER: <number>' and explain why "
    "this summary was selected in relation to the question.\n"
)
DEFAULT_QUERY_PROMPT = PromptTemplate(
    DEFAULT_QUERY_PROMPT_TMPL, prompt_type=PromptType.TREE_SELECT
)

# multiple choice
DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL = (
    "Some choices are given below. It is provided in a numbered "
    "list (1 to {num_chunks}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the top choices "
    "(no more than {branching_factor}, ranked by most relevant to least) that "
    "are most relevant to the question: '{query_str}'\n"
    "Provide choices in the following format: 'ANSWER: <numbers>' and explain why "
    "these summaries were selected in relation to the question.\n"
)
DEFAULT_QUERY_PROMPT_MULTIPLE = PromptTemplate(
    DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL, prompt_type=PromptType.TREE_SELECT_MULTIPLE
)


DEFAULT_REFINE_PROMPT_TMPL = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. "
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)
DEFAULT_REFINE_PROMPT = PromptTemplate(
    DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

DEFAULT_TREE_SUMMARIZE_TMPL = (
    "Context information from multiple sources is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from multiple sources and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
DEFAULT_TREE_SUMMARIZE_PROMPT = PromptTemplate(
    DEFAULT_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY
)


############################################
# Keyword Table
############################################

DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "Some text is provided below. Given the text, extract up to {max_keywords} "
    "keywords from the text. Avoid stopwords."
    "---------------------\n"
    "{text}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)
DEFAULT_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL, prompt_type=PromptType.KEYWORD_EXTRACT
)


# NOTE: the keyword extraction for queries can be the same as
# the one used to build the index, but here we tune it to see if performance is better.
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)
DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)


############################################
# Structured Store
############################################

DEFAULT_SCHEMA_EXTRACT_TMPL = (
    "We wish to extract relevant fields from an unstructured text chunk into "
    "a structured schema. We first provide the unstructured text, and then "
    "we provide the schema that we wish to extract. "
    "-----------text-----------\n"
    "{text}\n"
    "-----------schema-----------\n"
    "{schema}\n"
    "---------------------\n"
    "Given the text and schema, extract the relevant fields from the text in "
    "the following format: "
    "field1: <value>\nfield2: <value>\n...\n\n"
    "If a field is not present in the text, don't include it in the output."
    "If no fields are present in the text, return a blank string.\n"
    "Fields: "
)
DEFAULT_SCHEMA_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_SCHEMA_EXTRACT_TMPL, prompt_type=PromptType.SCHEMA_EXTRACT
)

# NOTE: taken from langchain and adapted
# https://github.com/langchain-ai/langchain/blob/v0.0.303/libs/langchain/langchain/chains/sql_database/prompt.py
DEFAULT_TEXT_TO_SQL_TMPL = (
    "Given an input question, first create a syntactically correct {dialect} "
    "query to run, then look at the results of the query and return the answer. "
    "You can order the results by a relevant column to return the most "
    "interesting examples in the database.\n\n"
    "Never query for all the columns from a specific table, only ask for a "
    "few relevant columns given the question.\n\n"
    "Pay attention to use only the column names that you can see in the schema "
    "description. "
    "Be careful to not query for columns that do not exist. "
    "Pay attention to which column is in which table. "
    "Also, qualify column names with the table name when needed. "
    "You are required to use the following format, each taking one line:\n\n"
    "Question: Question here\n"
    "SQLQuery: SQL Query to run\n"
    "SQLResult: Result of the SQLQuery\n"
    "Answer: Final answer here\n\n"
    "Only use tables listed below.\n"
    "{schema}\n\n"
    "Question: {query_str}\n"
    "SQLQuery: "
)

DEFAULT_TEXT_TO_SQL_PROMPT = PromptTemplate(
    DEFAULT_TEXT_TO_SQL_TMPL,
    prompt_type=PromptType.TEXT_TO_SQL,
)

DEFAULT_TEXT_TO_SQL_PGVECTOR_TMPL = """\
Given an input question, first create a syntactically correct {dialect} \
query to run, then look at the results of the query and return the answer. \
You can order the results by a relevant column to return the most \
interesting examples in the database.

Pay attention to use only the column names that you can see in the schema \
description. Be careful to not query for columns that do not exist. \
Pay attention to which column is in which table. Also, qualify column names \
with the table name when needed.

IMPORTANT NOTE: you can use specialized pgvector syntax (`<->`) to do nearest \
neighbors/semantic search to a given vector from an embeddings column in the table. \
The embeddings value for a given row typically represents the semantic meaning of that row. \
The vector represents an embedding representation \
of the question, given below. Do NOT fill in the vector values directly, but rather specify a \
`[query_vector]` placeholder. For instance, some select statement examples below \
(the name of the embeddings column is `embedding`):
SELECT * FROM items ORDER BY embedding <-> '[query_vector]' LIMIT 5;
SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 5;
SELECT * FROM items WHERE embedding <-> '[query_vector]' < 5;

You are required to use the following format, \
each taking one line:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use tables listed below.
{schema}


Question: {query_str}
SQLQuery: \
"""

DEFAULT_TEXT_TO_SQL_PGVECTOR_PROMPT = PromptTemplate(
    DEFAULT_TEXT_TO_SQL_PGVECTOR_TMPL,
    prompt_type=PromptType.TEXT_TO_SQL,
)


# NOTE: by partially filling schema, we can reduce to a QuestionAnswer prompt
# that we can feed to ur table
DEFAULT_TABLE_CONTEXT_TMPL = (
    "We have provided a table schema below. "
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "We have also provided context information below. "
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and the table schema, "
    "give a response to the following task: {query_str}"
)

DEFAULT_TABLE_CONTEXT_QUERY = (
    "Provide a high-level description of the table, "
    "as well as a description of each column in the table. "
    "Provide answers in the following format:\n"
    "TableDescription: <description>\n"
    "Column1Description: <description>\n"
    "Column2Description: <description>\n"
    "...\n\n"
)

DEFAULT_TABLE_CONTEXT_PROMPT = PromptTemplate(
    DEFAULT_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)

# NOTE: by partially filling schema, we can reduce to a refine prompt
# that we can feed to ur table
DEFAULT_REFINE_TABLE_CONTEXT_TMPL = (
    "We have provided a table schema below. "
    "---------------------\n"
    "{schema}\n"
    "---------------------\n"
    "We have also provided some context information below. "
    "{context_msg}\n"
    "---------------------\n"
    "Given the context information and the table schema, "
    "give a response to the following task: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)
DEFAULT_REFINE_TABLE_CONTEXT_PROMPT = PromptTemplate(
    DEFAULT_REFINE_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
)


############################################
# Knowledge-Graph Table
############################################

DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
    "---------------------\n"
    "Example:"
    "Text: Alice is Bob's mother."
    "Triplets:\n(Alice, is mother of, Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz, is, coffee shop)\n"
    "(Philz, founded in, Berkeley)\n"
    "(Philz, founded in, 1982)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)
DEFAULT_KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_KG_TRIPLET_EXTRACT_TMPL,
    prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT,
)

DEFAULT_DYNAMIC_EXTRACT_TMPL = (
    "Extract up to {max_knowledge_triplets} knowledge triplets from the given text. "
    "Each triplet should be in the form of (head, relation, tail) with their respective types.\n"
    "---------------------\n"
    "INITIAL ONTOLOGY:\n"
    "Entity Types: {allowed_entity_types}\n"
    "Relation Types: {allowed_relation_types}\n"
    "\n"
    "Use these types as a starting point, but introduce new types if necessary based on the context.\n"
    "\n"
    "GUIDELINES:\n"
    "- Output in JSON format: [{{'head': '', 'head_type': '', 'relation': '', 'tail': '', 'tail_type': ''}}]\n"
    "- Use the most complete form for entities (e.g., 'United States of America' instead of 'USA')\n"
    "- Keep entities concise (3-5 words max)\n"
    "- Break down complex phrases into multiple triplets\n"
    "- Ensure the knowledge graph is coherent and easily understandable\n"
    "---------------------\n"
    "EXAMPLE:\n"
    "Text: Tim Cook, CEO of Apple Inc., announced the new Apple Watch that monitors heart health. "
    "UC Berkeley researchers studied the benefits of apples.\n"
    "Output:\n"
    "[{{'head': 'Tim Cook', 'head_type': 'PERSON', 'relation': 'CEO_OF', 'tail': 'Apple Inc.', 'tail_type': 'COMPANY'}},\n"
    " {{'head': 'Apple Inc.', 'head_type': 'COMPANY', 'relation': 'PRODUCES', 'tail': 'Apple Watch', 'tail_type': 'PRODUCT'}},\n"
    " {{'head': 'Apple Watch', 'head_type': 'PRODUCT', 'relation': 'MONITORS', 'tail': 'heart health', 'tail_type': 'HEALTH_METRIC'}},\n"
    " {{'head': 'UC Berkeley', 'head_type': 'UNIVERSITY', 'relation': 'STUDIES', 'tail': 'benefits of apples', 'tail_type': 'RESEARCH_TOPIC'}}]\n"
    "---------------------\n"
    "Text: {text}\n"
    "Output:\n"
)

DEFAULT_DYNAMIC_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_DYNAMIC_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)

DEFAULT_DYNAMIC_EXTRACT_PROPS_TMPL = (
    "Extract up to {max_knowledge_triplets} knowledge triplets from the given text. "
    "Each triplet should be in the form of (head, relation, tail) with their respective types and properties.\n"
    "---------------------\n"
    "INITIAL ONTOLOGY:\n"
    "Entity Types: {allowed_entity_types}\n"
    "Entity Properties: {allowed_entity_properties}\n"
    "Relation Types: {allowed_relation_types}\n"
    "Relation Properties: {allowed_relation_properties}\n"
    "\n"
    "Use these types as a starting point, but introduce new types if necessary based on the context.\n"
    "\n"
    "GUIDELINES:\n"
    "- Output in JSON format: [{{'head': '', 'head_type': '', 'head_props': {{...}}, 'relation': '', 'relation_props': {{...}}, 'tail': '', 'tail_type': '', 'tail_props': {{...}}}}]\n"
    "- Use the most complete form for entities (e.g., 'United States of America' instead of 'USA')\n"
    "- Keep entities concise (3-5 words max)\n"
    "- Break down complex phrases into multiple triplets\n"
    "- Ensure the knowledge graph is coherent and easily understandable\n"
    "---------------------\n"
    "EXAMPLE:\n"
    "Text: Tim Cook, CEO of Apple Inc., announced the new Apple Watch that monitors heart health. "
    "UC Berkeley researchers studied the benefits of apples.\n"
    "Output:\n"
    "[{{'head': 'Tim Cook', 'head_type': 'PERSON', 'head_props': {{'prop1': 'val', ...}}, 'relation': 'CEO_OF', 'relation_props': {{'prop1': 'val', ...}}, 'tail': 'Apple Inc.', 'tail_type': 'COMPANY', 'tail_props': {{'prop1': 'val', ...}}}},\n"
    " {{'head': 'Apple Inc.', 'head_type': 'COMPANY', 'head_props': {{'prop1': 'val', ...}}, 'relation': 'PRODUCES', 'relation_props': {{'prop1': 'val', ...}}, 'tail': 'Apple Watch', 'tail_type': 'PRODUCT', 'tail_props': {{'prop1': 'val', ...}}}},\n"
    " {{'head': 'Apple Watch', 'head_type': 'PRODUCT', 'head_props': {{'prop1': 'val', ...}}, 'relation': 'MONITORS', 'relation_props': {{'prop1': 'val', ...}}, 'tail': 'heart health', 'tail_type': 'HEALTH_METRIC', 'tail_props': {{'prop1': 'val', ...}}}},\n"
    " {{'head': 'UC Berkeley', 'head_type': 'UNIVERSITY', 'head_props': {{'prop1': 'val', ...}}, 'relation': 'STUDIES', 'relation_props': {{'prop1': 'val', ...}}, 'tail': 'benefits of apples', 'tail_type': 'RESEARCH_TOPIC', 'tail_props': {{'prop1': 'val', ...}}}}]\n"
    "---------------------\n"
    "Text: {text}\n"
    "Output:\n"
)

DEFAULT_DYNAMIC_EXTRACT_PROPS_PROMPT = PromptTemplate(
    DEFAULT_DYNAMIC_EXTRACT_PROPS_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)

############################################
# HYDE
##############################################

HYDE_TMPL = (
    "Please write a passage to answer the question\n"
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'Passage:"""\n'
)

DEFAULT_HYDE_PROMPT = PromptTemplate(HYDE_TMPL, prompt_type=PromptType.SUMMARY)


############################################
# Simple Input
############################################

DEFAULT_SIMPLE_INPUT_TMPL = "{query_str}"
DEFAULT_SIMPLE_INPUT_PROMPT = PromptTemplate(
    DEFAULT_SIMPLE_INPUT_TMPL, prompt_type=PromptType.SIMPLE_INPUT
)


############################################
# JSON Path
############################################

DEFAULT_JSON_PATH_TMPL = (
    "We have provided a JSON schema below:\n"
    "{schema}\n"
    "Given a task, respond with a JSON Path query that "
    "can retrieve data from a JSON value that matches the schema.\n"
    "Provide the JSON Path query in the following format: 'JSONPath: <JSONPath>'\n"
    "You must include the value 'JSONPath:' before the provided JSON Path query."
    "Example Format:\n"
    "Task: What is John's age?\n"
    "Response: JSONPath: $.John.age\n"
    "Let's try this now: \n\n"
    "Task: {query_str}\n"
    "Response: "
)

DEFAULT_JSON_PATH_PROMPT = PromptTemplate(
    DEFAULT_JSON_PATH_TMPL, prompt_type=PromptType.JSON_PATH
)


############################################
# Choice Select
############################################

DEFAULT_CHOICE_SELECT_PROMPT_TMPL = (
    "A list of documents is shown below. Each document has a number next to it along "
    "with a summary of the document. A question is also provided. \n"
    "Respond with the numbers of the documents "
    "you should consult to answer the question, in order of relevance, as well \n"
    "as the relevance score. The relevance score is a number from 1-10 based on "
    "how relevant you think the document is to the question.\n"
    "Do not include any documents that are not relevant to the question. \n"
    "Example format: \n"
    "Document 1:\n<summary of document 1>\n\n"
    "Document 2:\n<summary of document 2>\n\n"
    "...\n\n"
    "Document 10:\n<summary of document 10>\n\n"
    "Question: <question>\n"
    "Answer:\n"
    "Doc: 9, Relevance: 7\n"
    "Doc: 3, Relevance: 4\n"
    "Doc: 7, Relevance: 3\n\n"
    "Let's try this now: \n\n"
    "{context_str}\n"
    "Question: {query_str}\n"
    "Answer:\n"
)
DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(
    DEFAULT_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)


############################################
# RankGPT Rerank template
############################################

RANKGPT_RERANK_PROMPT_TMPL = (
    "Search Query: {query}. \nRank the {num} passages above "
    "based on their relevance to the search query. The passages "
    "should be listed in descending order using identifiers. "
    "The most relevant passages should be listed first. "
    "The output format should be [] > [], e.g., [1] > [2]. "
    "Only response the ranking results, "
    "do not say any word or explain."
)
RANKGPT_RERANK_PROMPT = PromptTemplate(
    RANKGPT_RERANK_PROMPT_TMPL, prompt_type=PromptType.RANKGPT_RERANK
)


############################################
# JSONalyze Query Template
############################################

DEFAULT_JSONALYZE_PROMPT_TMPL = (
    "You are given a table named: '{table_name}' with schema, "
    "generate SQLite SQL query to answer the given question.\n"
    "Table schema:\n"
    "{table_schema}\n"
    "Question: {question}\n\n"
    "SQLQuery: "
)

DEFAULT_JSONALYZE_PROMPT = PromptTemplate(
    DEFAULT_JSONALYZE_PROMPT_TMPL, prompt_type=PromptType.TEXT_TO_SQL
)

###########################################
# REBEL MetaPrompt Template
###########################################

DEFAULT_REBEL_META_PROMPT_TMPL = (
    '''You are a prompt generator. You will receive only a user's query as input. Your task is to:

Analyze the user's query and identify additional properties beyond basic relevance that would be desirable for selecting and ranking context documents. These properties should be inferred from the query's subject matter, without the user specifying them. Such properties may include:

Domain appropriateness (e.g., technical accuracy, authoritative sourcing, correctness of information)
Perspective diversity (multiple viewpoints, ideological balance, different theoretical frameworks)
Temporal relevance (up-to-date information, recent data)
Depth of detail and specificity (thorough coverage, multi-faceted analysis, detailed examples)
Trustworthiness, neutrality, impartiality (reliable sources, unbiased accounts)
Reasoning depth or conceptual complexity
Authoritativeness (recognition of reputable experts, institutions, or high-quality sources)
After inferring these properties from the query, produce a final prompt that instructs a large-language model re-ranker on how to:

Take the user's query and a set of candidate documents.
The documents and the query will appear after your instructions in this format: A list of documents is shown below. Each document has a number and a summary. The summaries may indicate the type of source, credibility level, publication date, or the nature of the information. After listing all documents, the user's query will be presented on a single line labeled "Question:". For example: Document 1: <summary of document 1> Document 2: <summary of document 2> ... Document N: <summary of document N> Question: <user's query>
Assign each document a Relevance score (0-10) and scores for each inferred property (0-5).
Compute a weighted composite score for each document. This composite score should not just be used to break ties, but to determine the final ordering. For instance, you may define a formula like: Final Score = Relevance + (Weight1 * Property1) + (Weight2 * Property2) + ... The weights should be specified by you. For example, if you have three properties, you might say: Final Score = Relevance + 0.5*(Property1) + 0.5*(Property2) + 0.5*(Property3) This ensures that documents which strongly exhibit the desired secondary properties can surpass documents with slightly higher relevance but weaker secondary property scores.
Filter out irrelevant documents first. For example, discard any document with Relevance < 3.
Rank all remaining documents by their Final Score (based on the chosen weights).
If two documents end up with the exact same Final Score, you may choose a consistent approach to pick one over the other (e.g., prefer the document with higher authoritativeness).
If no documents meet the relevance threshold, output nothing.
Produce only the final ranked list of chosen documents with their Final Score, in descending order of Final Score. The format for each selected document should be: Doc: [document number], Relevance: [score], where [score] is actually the final score - not the relevance score.
Include no commentary, explanation, or additional text beyond these lines.
Your final prompt should:

Include the user's query verbatim.
Enumerate and define the inferred properties in detail, clearly stating their significance.
Provide the exact scoring rubric for Relevance (0-10) and each inferred property (0-5), explaining what high and low scores mean.
Specify the weighted composite score formula and list the weights for each property.
Give a step-by-step procedure: assign Relevance, assign property scores, discard low-relevance documents, compute Final Scores, sort by Final Score, handle ties if any, then output the final list.
State what to do if no documents qualify (output nothing).
Remind the re-ranker that the documents and query will be shown after this prompt, and that the only acceptable output is the final sorted list of documents and their relevance scores.
Your output should be a single prompt that can be given directly to the large-language model re-ranker. After this prompt, the re-ranker will receive the documents and the query and must follow the instructions to produce the final answer.

At the end of your prompt, you should ALWAYS NO MATTER WHAT include the following:

"Example format: \n"
"Document 1:\n<summary of document 1>\n\n"
"Document 2:\n<summary of document 2>\n\n"
"...\n\n"
"Document 10:\n<summary of document 10>\n\n"
"Question: <question>\n"
"Answer:\n"
"Doc: 9, Relevance: 7\n"
"Doc: 3, Relevance: 4\n"
"Doc: 7, Relevance: 3\n\n"
"Let's try this now: \n\n"
"{context_str}\n"
"Question: {query_str}\n"
"Answer:\n"

Below are 5 k-shot examples demonstrating the required level of detail and explicitness. Each example:

Presents a user query.
Infers multiple properties and explains their relevance.
Provides a scoring rubric for Relevance and the inferred properties.
Defines a weighted composite scoring formula that incorporates Relevance and all secondary properties.
Gives step-by-step instructions for scoring, filtering, ranking, and outputting results.
Explains what to do if no suitable documents remain.
Instructs that the final output should only be lines of the form "Doc: [number], Relevance: [score]" with no extra text.
Example 1 User Query: "How do different countries' tax policies affect income inequality, and what arguments exist from various economic schools of thought?"

Inferred Properties:

Perspective diversity (0-5): Documents that mention or compare multiple economic theories or viewpoints score higher. A high score (5) means it covers several distinct schools of thought. A low score (0) means it is one-dimensional.
Authoritativeness (0-5): Documents from credible economists, reputable research institutes, or peer-reviewed studies score higher. A 5 might be a well-cited academic paper; a 0 might be an anonymous blog post.
Comparative breadth (0-5): Documents discussing tax policies in multiple countries score higher. A 5 means it covers several countries, a 0 means it focuses on just one or does not compare countries at all.
Scoring Rubric: Relevance (0-10): A 10 means the document directly addresses how tax policies influence income inequality and references arguments from different economic viewpoints. A 0 means it is off-topic. Perspective diversity (0-5): Assign based on how many distinct economic perspectives are included. Authoritativeness (0-5): Assign based on credibility and source quality. Comparative breadth (0-5): Assign based on the number of countries or breadth of international comparison.

Weighted Composite Score: Final Score = Relevance + 0.5*(Perspective diversity) + 0.5*(Authoritativeness) + 0.5*(Comparative breadth)

Instructions: After this prompt, you will see: Document 1: <summary> Document 2: <summary> ... Document N: <summary> Question: "How do different countries' tax policies affect income inequality, and what arguments exist from various economic schools of thought?"

Assign Relevance to each document (0-10). Discard documents with Relevance < 3.
For remaining documents, assign Perspective diversity, Authoritativeness, and Comparative breadth (each 0-5).
Compute Final Score as described above.
Sort all remaining documents by Final Score (descending).
If two documents have identical Final Scores, pick consistently, for example by preferring the one with higher Authoritativeness.
If no document remains, output nothing.
Output only: Doc: [number], Relevance: [score] for each selected document, no commentary or explanation, where [score] is actually the final score.

"Example format: \n"
"Document 1:\n<summary of document 1>\n\n"
"Document 2:\n<summary of document 2>\n\n"
"...\n\n"
"Document 10:\n<summary of document 10>\n\n"
"Question: <question>\n"
"Answer:\n"
"Doc: 9, Relevance: 7\n"
"Doc: 3, Relevance: 4\n"
"Doc: 7, Relevance: 3\n\n"
"Let's try this now: \n\n"
"{context_str}\n"
"Question: {query_str}\n"
"Answer:\n"


Example 2 User Query: "What are the latest recommended treatments for chronic lower back pain according to recent medical research?"

Inferred Properties:

Recency (0-5): Higher if the document references recent studies, new clinical guidelines, or up-to-date research (within the last few years). A 5 means it is very recent, a 0 means outdated or no mention of timeliness.
Authoritativeness (0-5): Higher if sourced from reputable medical journals, recognized health organizations, or consensus guidelines.
Specificity (0-5): Higher if it focuses specifically on chronic lower back pain treatments. A 5 means it precisely addresses chronic lower back pain, a 0 means it only vaguely mentions pain or general treatments without specificity.
Scoring Rubric: Relevance (0-10): A 10 means the document explicitly discusses current recommended treatments for chronic lower back pain based on recent research. A 0 means off-topic. Recency (0-5) Authoritativeness (0-5) Specificity (0-5)

Weighted Composite Score: Final Score = Relevance + 0.5*(Recency) + 0.5*(Authoritativeness) + 0.5*(Specificity)

Instructions: After this prompt: Document 1: <summary> ... Document N: <summary> Question: "What are the latest recommended treatments for chronic lower back pain according to recent medical research?"

Assign Relevance. Exclude Relevance < 3.
Assign Recency, Authoritativeness, Specificity.
Compute Final Score.
Sort by Final Score.
If tied, choose consistently (e.g., prefer higher Authoritativeness).
If none remain, output nothing.
Output only lines like: Doc: X, Relevance: Y, where Y is actually the final score.

"Example format: \n"
"Document 1:\n<summary of document 1>\n\n"
"Document 2:\n<summary of document 2>\n\n"
"...\n\n"
"Document 10:\n<summary of document 10>\n\n"
"Question: <question>\n"
"Answer:\n"
"Doc: 9, Relevance: 7\n"
"Doc: 3, Relevance: 4\n"
"Doc: 7, Relevance: 3\n\n"
"Let's try this now: \n\n"
"{context_str}\n"
"Question: {query_str}\n"
"Answer:\n"


Example 3 User Query: "How did the policies of Emperor Qin Shi Huang shape the political and cultural landscape of ancient China?"

Inferred Properties:

Historical depth (0-5): Higher if it provides detailed historical context, dates, and direct evidence. A 5 is richly detailed, a 0 is very superficial.
Perspective range (0-5): Higher if it references multiple historians or scholarly opinions. A 5 means multiple perspectives, a 0 is one-sided.
Cultural/political detail (0-5): Higher if it addresses both political structures and cultural changes. A 5 is comprehensive, a 0 is minimal detail.
Scoring Rubric: Relevance (0-10): A 10 means it explicitly discusses Qin Shi Huang's policies and their impact on both political and cultural aspects of ancient China. Historical depth (0-5) Perspective range (0-5) Cultural/political detail (0-5)

Weighted Composite Score: Final Score = Relevance + 0.5*(Historical depth) + 0.5*(Perspective range) + 0.5*(Cultural/political detail)

Instructions: After this prompt: Document 1: <summary> ... Document N: <summary> Question: "How did the policies of Emperor Qin Shi Huang shape the political and cultural landscape of ancient China?"

Assign Relevance, discard < 3.
Assign Historical depth, Perspective range, Cultural/political detail.
Compute Final Score.
Sort by Final Score.
Tie-break by preferring more historically authoritative perspectives if still tied.
If none qualify, output nothing.
Only output: Doc: [number], Relevance: [score], where [score] is actually the final score.

"Example format: \n"
"Document 1:\n<summary of document 1>\n\n"
"Document 2:\n<summary of document 2>\n\n"
"...\n\n"
"Document 10:\n<summary of document 10>\n\n"
"Question: <question>\n"
"Answer:\n"
"Doc: 9, Relevance: 7\n"
"Doc: 3, Relevance: 4\n"
"Doc: 7, Relevance: 3\n\n"
"Let's try this now: \n\n"
"{context_str}\n"
"Question: {query_str}\n"
"Answer:\n"


Example 4 User Query: "What are the main differences between various machine learning frameworks like TensorFlow, PyTorch, and Scikit-learn?"

Inferred Properties:

Technical accuracy (0-5): Higher if the document correctly and specifically describes features, performance characteristics, or typical uses. A 5 means very accurate and specific.
Comparative breadth (0-5): Higher if the document compares multiple frameworks directly, ideally all three. A 5 means it covers all three well, a 0 means it only mentions one.
Authoritativeness (0-5): Higher if citing official documentation, known ML experts, or reputable evaluation sources.
Scoring Rubric: Relevance (0-10): A 10 means the document explicitly compares these ML frameworks in detail. Technical accuracy (0-5) Comparative breadth (0-5) Authoritativeness (0-5)

Weighted Composite Score: Final Score = Relevance + 0.5*(Technical accuracy) + 0.5*(Comparative breadth) + 0.5*(Authoritativeness)

Instructions: After prompt: Document 1: <summary> ... Document N: <summary> Question: "What are the main differences between various machine learning frameworks like TensorFlow, PyTorch, and Scikit-learn?"

Assign Relevance, exclude < 3.
Assign Technical accuracy, Comparative breadth, Authoritativeness.
Compute Final Score.
Sort by Final Score.
Tie-break by preferring documents that are more authoritative or have greater comparative breadth.
If none remain, output nothing.
Output only lines like: Doc: [number], Relevance: [score], where [score] is actually the final score.

"Example format: \n"
"Document 1:\n<summary of document 1>\n\n"
"Document 2:\n<summary of document 2>\n\n"
"...\n\n"
"Document 10:\n<summary of document 10>\n\n"
"Question: <question>\n"
"Answer:\n"
"Doc: 9, Relevance: 7\n"
"Doc: 3, Relevance: 4\n"
"Doc: 7, Relevance: 3\n\n"
"Let's try this now: \n\n"
"{context_str}\n"
"Question: {query_str}\n"
"Answer:\n"

Example 5 User Query: "What are the arguments for and against universal basic income in modern economies?"

Inferred Properties:

Balance of perspectives (0-5): Higher if the document presents both pro and con arguments. A 5 means thorough coverage of both sides.
Reasoning depth (0-5): Higher if it explains the rationale behind arguments, providing logic or evidence.
Authoritativeness (0-5): Higher if referencing economists, studies, or policy analyses from reputable sources.
Scoring Rubric: Relevance (0-10): A 10 means it clearly discusses UBI arguments both for and against. Balance of perspectives (0-5) Reasoning depth (0-5) Authoritativeness (0-5)

Weighted Composite Score: Final Score = Relevance + 0.5*(Balance of perspectives) + 0.5*(Reasoning depth) + 0.5*(Authoritativeness)

Instructions: After prompt: Document 1: <summary> ... Document N: <summary> Question: "What are the arguments for and against universal basic income in modern economies?"

Assign Relevance, discard < 3.
Assign Balance of perspectives, Reasoning depth, Authoritativeness.
Compute Final Score.
Sort by Final Score.
If tied, prefer documents with higher reasoning depth or authoritativeness.
If none remain, output nothing.
Output only: Doc: [number], Relevance: [score], where [score] is actually the final score.

"Example format: \n"
"Document 1:\n<summary of document 1>\n\n"
"Document 2:\n<summary of document 2>\n\n"
"...\n\n"
"Document 10:\n<summary of document 10>\n\n"
"Question: <question>\n"
"Answer:\n"
"Doc: 9, Relevance: 7\n"
"Doc: 3, Relevance: 4\n"
"Doc: 7, Relevance: 3\n\n"
"Let's try this now: \n\n"
"{context_str}\n"
"Question: {query_str}\n"
"Answer:\n"


Follow these examples as a template for your final prompt. For any new user query, do the following:

Include the user's query verbatim.
Infer the relevant secondary properties and define them clearly.
Give a scoring rubric for Relevance and each property.
Specify a weighted composite score formula that combines Relevance and the properties.
Provide step-by-step instructions: assign scores, filter out irrelevant documents, compute Final Score, sort by Final Score, handle ties, and if none qualify, output nothing.
Instruct the re-ranker to output only the final list of documents and their Relevance scores, with no extra commentary.
Now, here is the user's query:

{user_query}
'''
)

DEFAULT_REBEL_META_PROMPT = PromptTemplate(
    DEFAULT_REBEL_META_PROMPT_TMPL, prompt_type=PromptType.REBEL_RERANK
)

DEFAULT_REBEL_CHOICE_SELECT_PROMPT_TMPL = (
"""You are a re-ranking system. Your task is to analyze a user's query and a set of candidate documents, assign scores based on specified properties, and output the final ranking of documents.

**Inferred Properties**

1. **Depth of Content (0-5):**
- Higher scores indicate thorough detail and comprehensive coverage of the topic.
- A "5" is exceptionally in-depth with multiple facets addressed; a "0" is very superficial.

2. **Diversity of Perspectives (0-5):**
- Higher scores indicate that multiple viewpoints or angles are represented.
- A "5" means it engages with a variety of perspectives or sources; a "0" means it is entirely one-sided.

3. **Clarity and Specificity (0-5):**
- Higher scores indicate that the document presents information clearly and addresses the query with precise, unambiguous detail.
- A "5" means it is highly specific and clear, while a "0" means it is vague or overly general.

4. **Authoritativeness (0-5):**
- Higher scores indicate reputable sources, expert authorship, or recognized credibility.
- A "5" might be an extensively cited academic work or an official standard; a "0" would be an unknown or dubious source.

5. **Recency (0-5):**
- Higher scores indicate that the document references recent studies, data, or developments.
- A "5" means it is very current and up-to-date; a "0" means it is outdated or does not reference any time-sensitive information.

**Scoring Rubric**

- **Relevance (0-10):**
- A "10" means the document directly addresses the user's query, covering the key subject comprehensively.
- A "0" means it is completely off-topic.

- **Depth of Content (0-5):** Based on how detailed or thorough the document is.
- **Diversity of Perspectives (0-5):** Based on how many viewpoints or angles are presented.
- **Clarity and Specificity (0-5):** Based on how clear and precise the document is.
- **Authoritativeness (0-5):** Based on source credibility or recognized expertise.
- **Recency (0-5):** Based on how up-to-date the document is.

**Weighted Composite Score**
Final Score = Relevance + 0.5*(Depth of Content) + 0.5*(Diversity of Perspectives) + 0.5*(Clarity and Specificity) + 0.5*(Authoritativeness) + 0.5*(Recency)

**Instructions**
1. Assign Relevance to each document on a scale of 0-10. Discard any document with Relevance < 3.
2. For the remaining documents, assign scores for:
- Depth of Content (0-5)
- Diversity of Perspectives (0-5)
- Clarity and Specificity (0-5)
- Authoritativeness (0-5)
- Recency (0-5)
3. Compute each document's Final Score using the formula above.
4. Sort the documents by their Final Score in descending order.
5. If two documents end up with the same Final Score, prefer the one with higher Authoritativeness (or apply another consistent tie-breaking rule).
6. If no documents remain after filtering for Relevance, output nothing.
7. Output only the list of theselected documents with their Relevance scores, in this format (no extra text or commentary), where [score] is actually the Final Score and NOT the relevance score.
```
Doc: [document number], Relevance: [score]
```

**Example format:**
```
Document 1:
<summary of document 1>

Document 2:
<summary of document 2>

...

Document 10:
<summary of document 10>

Question: <question>
Answer:
Doc: 9, Relevance: 7
Doc: 3, Relevance: 4
Doc: 7, Relevance: 3

Let's try this now:

{context_str}
Question: {query_str}
Answer:
```""")

DEFAULT_REBEL_CHOICE_SELECT_PROMPT = PromptTemplate(
    DEFAULT_REBEL_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)
