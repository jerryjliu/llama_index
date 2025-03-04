from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from llama_index.core.callbacks.pythonically_printing_base_handler import (
    PythonicallyPrintingBaseHandler,
)
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.utilities.token_counting import TokenCounter
from llama_index.core.utils import get_tokenizer
import logging

if TYPE_CHECKING:
    from llama_index.core.llms import ChatResponse, CompletionResponse


@dataclass
class TokenCountingEvent:
    prompt: str
    completion: str
    completion_token_count: int
    prompt_token_count: int
    total_token_count: int = 0
    cached_tokens: int = 0 
    event_id: str = ""

    def __post_init__(self) -> None:
        self.total_token_count = self.prompt_token_count + self.completion_token_count


def get_tokens_from_response(
    response: Union["CompletionResponse", "ChatResponse"]
) -> Tuple[int, int]:
    """Get the token counts from a raw response."""
    raw_response = response.raw
    if not isinstance(raw_response, dict):
        raw_response = dict(raw_response or {})

    usage = raw_response.get("usage", {})
    if usage is None:
        usage = response.additional_kwargs

    if not usage:
        return 0, 0

    if not isinstance(usage, dict):
        usage = usage.model_dump()

    possible_input_keys = ("prompt_tokens", "input_tokens")
    possible_output_keys = ("completion_tokens", "output_tokens")
    openai_prompt_tokens_details_key = 'prompt_tokens_details'
    
    prompt_tokens = 0
    for input_key in possible_input_keys:
        if input_key in usage:
            prompt_tokens = usage[input_key]
            break

    completion_tokens = 0
    for output_key in possible_output_keys:
        if output_key in usage:
            completion_tokens = usage[output_key]
            break
        
    cached_tokens = 0
    if openai_prompt_tokens_details_key in usage:
        cached_tokens = usage[openai_prompt_tokens_details_key]['cached_tokens']  
        
    return prompt_tokens, completion_tokens, cached_tokens 


def get_llm_token_counts(
    token_counter: TokenCounter, payload: Dict[str, Any], event_id: str = ""
) -> TokenCountingEvent:
    from llama_index.core.llms import ChatMessage

    if EventPayload.PROMPT in payload:
        prompt = payload.get(EventPayload.PROMPT)
        completion = payload.get(EventPayload.COMPLETION)

        if completion:
            # get from raw or additional_kwargs
            prompt_tokens, completion_tokens, cached_tokens = get_tokens_from_response(completion)
        else:
            prompt_tokens, completion_tokens, cached_tokens = 0, 0, 0

        if prompt_tokens == 0:
            prompt_tokens = token_counter.get_string_tokens(str(prompt))

        if completion_tokens == 0:
            completion_tokens = token_counter.get_string_tokens(str(completion))

        return TokenCountingEvent(
            event_id=event_id,
            prompt=str(prompt),
            prompt_token_count=prompt_tokens,
            completion=str(completion),
            completion_token_count=completion_tokens,
            cached_tokens=cached_tokens,
        )

    elif EventPayload.MESSAGES in payload:
        messages = cast(List[ChatMessage], payload.get(EventPayload.MESSAGES, []))
        messages_str = "\n".join([str(x) for x in messages])

        response = payload.get(EventPayload.RESPONSE)
        response_str = str(response)

        if response:
            prompt_tokens, completion_tokens, cached_tokens = get_tokens_from_response(response)
        else:
            prompt_tokens, completion_tokens, cached_tokens = 0, 0, 0

        if prompt_tokens == 0:
            prompt_tokens = token_counter.estimate_tokens_in_messages(messages)

        if completion_tokens == 0:
            completion_tokens = token_counter.get_string_tokens(response_str)

        return TokenCountingEvent(
            event_id=event_id,
            prompt=messages_str,
            prompt_token_count=prompt_tokens,
            completion=response_str,
            completion_token_count=completion_tokens,
            cached_tokens=cached_tokens,
        )
    else:
        return TokenCountingEvent(
            event_id=event_id,
            prompt="",
            prompt_token_count=0,
            completion="",
            completion_token_count=0,
            cached_tokens=0,
        )


class TokenCountingHandler(PythonicallyPrintingBaseHandler):
    """Callback handler for counting tokens in LLM and Embedding events.

    Args:
        tokenizer:
            Tokenizer to use. Defaults to the global tokenizer
            (see llama_index.core.utils.globals_helper).
        event_starts_to_ignore: List of event types to ignore at the start of a trace.
        event_ends_to_ignore: List of event types to ignore at the end of a trace.
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List]] = None,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.llm_token_counts: List[TokenCountingEvent] = []
        self.embedding_token_counts: List[TokenCountingEvent] = []
        self.tokenizer = tokenizer or get_tokenizer()

        self._token_counter = TokenCounter(tokenizer=self.tokenizer)
        self._verbose = verbose

        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
            logger=logger,
        )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        return

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        return

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Count the LLM or Embedding tokens as needed."""
        if (
            event_type == CBEventType.LLM
            and event_type not in self.event_ends_to_ignore
            and payload is not None
        ):
            self.llm_token_counts.append(
                get_llm_token_counts(
                    token_counter=self._token_counter,
                    payload=payload,
                    event_id=event_id,
                )
            )

            if self._verbose:
                self._print(
                    "LLM Prompt Token Usage: "
                    f"{self.llm_token_counts[-1].prompt_token_count}\n"
                    "LLM Completion Token Usage: "
                    f"{self.llm_token_counts[-1].completion_token_count}"
                    "LLM Cached Tokens: "
                    f"{self.llm_token_counts[-1].cached_tokens}",
                )
        elif (
            event_type == CBEventType.EMBEDDING
            and event_type not in self.event_ends_to_ignore
            and payload is not None
        ):
            total_chunk_tokens = 0
            for chunk in payload.get(EventPayload.CHUNKS, []):
                self.embedding_token_counts.append(
                    TokenCountingEvent(
                        event_id=event_id,
                        prompt=chunk,
                        prompt_token_count=self._token_counter.get_string_tokens(chunk),
                        completion="",
                        completion_token_count=0,
                    )
                )
                total_chunk_tokens += self.embedding_token_counts[-1].total_token_count

            if self._verbose:
                self._print(f"Embedding Token Usage: {total_chunk_tokens}")

    @property
    def total_llm_token_count(self) -> int:
        """Get the current total LLM token count."""
        return sum([x.total_token_count for x in self.llm_token_counts])

    @property
    def prompt_llm_token_count(self) -> int:
        """Get the current total LLM prompt token count."""
        return sum([x.prompt_token_count for x in self.llm_token_counts])

    @property
    def completion_llm_token_count(self) -> int:
        """Get the current total LLM completion token count."""
        return sum([x.completion_token_count for x in self.llm_token_counts])
    
    @property
    def total_cached_token_count(self) -> int:
        """Get the current total cached token count."""
        return sum([x.cached_tokens for x in self.llm_token_counts])

    @property
    def total_embedding_token_count(self) -> int:
        """Get the current total Embedding token count."""
        return sum([x.total_token_count for x in self.embedding_token_counts])

    def reset_counts(self) -> None:
        """Reset the token counts."""
        self.llm_token_counts = []
        self.embedding_token_counts = []
