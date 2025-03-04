"""Prompt types enum."""

from enum import Enum


class PromptType(str, Enum):
    """Prompt type."""

    # summarization
    SUMMARY = "summary"
    # tree insert node
    TREE_INSERT = "insert"
    # tree select query prompt
    TREE_SELECT = "tree_select"
    # tree select query prompt (multiple)
    TREE_SELECT_MULTIPLE = "tree_select_multiple"
    # question-answer
    QUESTION_ANSWER = "text_qa"
    # refine
    REFINE = "refine"
    # keyword extract
    KEYWORD_EXTRACT = "keyword_extract"
    # query keyword extract
    QUERY_KEYWORD_EXTRACT = "query_keyword_extract"

    # schema extract
    SCHEMA_EXTRACT = "schema_extract"

    # text to sql
    TEXT_TO_SQL = "text_to_sql"

    # text to graph query
    TEXT_TO_GRAPH_QUERY = "text_to_graph_query"

    # table context
    TABLE_CONTEXT = "table_context"

    # KG extraction prompt
    KNOWLEDGE_TRIPLET_EXTRACT = "knowledge_triplet_extract"

    # Simple Input prompt
    SIMPLE_INPUT = "simple_input"

    # Pandas prompt
    PANDAS = "pandas"

    # JSON path prompt
    JSON_PATH = "json_path"

    # Single select prompt
    SINGLE_SELECT = "single_select"

    # Multiple select prompt
    MULTI_SELECT = "multi_select"

    VECTOR_STORE_QUERY = "vector_store_query"

    # Sub question prompt
    SUB_QUESTION = "sub_question"

    # SQL response synthesis prompt
    SQL_RESPONSE_SYNTHESIS = "sql_response_synthesis"

    # SQL response synthesis prompt (v2)
    SQL_RESPONSE_SYNTHESIS_V2 = "sql_response_synthesis_v2"

    # Conversation
    CONVERSATION = "conversation"

    # Decompose query transform
    DECOMPOSE = "decompose"

    # Choice select
    CHOICE_SELECT = "choice_select"

    # custom (by default)
    CUSTOM = "custom"

    # RankGPT rerank
    RANKGPT_RERANK = "rankgpt_rerank"

    # REBEL rerank
    REBEL_RERANK = "rebel_rerank"
